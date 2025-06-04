# tetris_rl_agents/agents/a2c_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
import os
import copy

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.tetris import Tetris

import config as global_config
from .base_agent import BaseAgent

DEVICE = global_config.DEVICE

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, seed=0,
                 fc1_units=global_config.A2C_FC1_UNITS,
                 fc2_units=global_config.A2C_FC2_UNITS):
        super(ActorCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Shared layers
        self.fc_shared1 = nn.Linear(state_size, fc1_units)
        self.fc_shared2 = nn.Linear(fc1_units, fc2_units)

        # Actor head: Takes shared features, outputs score for a *potential next state*
        # Its input features will be those from get_next_states()
        self.actor_head = nn.Linear(fc2_units, 1) 

        # Critic head: Takes shared features, outputs value for *current board state*
        # Its input features will be the board state *before* action selection
        self.critic_head = nn.Linear(fc2_units, 1)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state_features_input, is_actor_pass=False):
        """
        Forward pass.
        If is_actor_pass is True, state_features_input are features of potential next states (batch).
        If is_actor_pass is False, state_features_input is feature of a current state (for critic).
        """
        x = F.relu(self.fc_shared1(state_features_input))
        x_shared = F.relu(self.fc_shared2(x))
        
        if is_actor_pass:
            return self.actor_head(x_shared) # Scores for (batch of) potential next states
        else:
            return self.critic_head(x_shared) # Value for (a single or batch of) current state(s)


class A2CAgent(BaseAgent):
    def __init__(self, state_size, seed=0,
                 learning_rate=global_config.A2C_LEARNING_RATE,
                 gamma=global_config.A2C_GAMMA,
                 entropy_coeff=global_config.A2C_ENTROPY_COEFF,
                 value_loss_coeff=global_config.A2C_VALUE_LOSS_COEFF):
        super().__init__(state_size)
        self._agent_seed = seed
        random.seed(self._agent_seed)
        torch.manual_seed(self._agent_seed)
        if DEVICE.type == 'cuda':
            torch.cuda.manual_seed_all(self._agent_seed)

        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff

        self.network = ActorCriticNetwork(state_size, seed=self._agent_seed).to(DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # For on-policy updates, A2C typically doesn't need a large replay buffer.
        # It learns from the (state, action, reward, next_state) tuple directly.
        # However, some A2C variants (like A3C or N-step A2C) collect trajectories.
        # This basic A2C will learn per piece placement.

        print(f"A2C Agent initialized. LR: {learning_rate}, Gamma: {gamma}, Device: {DEVICE}")

    def select_action(self, current_board_features: torch.Tensor, tetris_game_instance: Tetris, epsilon_override=None):
        """
        Args:
            current_board_features (torch.Tensor): Features of the board *before* current piece placement.
                                                   Used by critic if needed during selection, but mainly for learning.
            tetris_game_instance (Tetris): Game object to get next possible states.
        Returns:
            action_tuple (tuple): (x_pos, rotation_idx)
            aux_info (dict): Contains 'log_prob', 'entropy', 'features_after_chosen_action', 'value_of_current_board'
        """
        next_steps_dict = tetris_game_instance.get_next_states()
        if not next_steps_dict: # Should be rare
            return (tetris_game_instance.width // 2, 0), {} 

        possible_actions_tuples = list(next_steps_dict.keys())
        # feature_tensors_for_next_steps are features of S' (if action taken)
        feature_tensors_for_next_steps = torch.stack(list(next_steps_dict.values())).to(DEVICE)

        self.network.eval() # Eval mode for selection
        with torch.no_grad():
            # Actor pass: get scores for all potential next states S'
            action_scores = self.network(feature_tensors_for_next_steps, is_actor_pass=True).squeeze(-1)
            # Critic pass: get value V(S_current)
            # Note: current_board_features is for the board state *before* placing the current piece.
            value_of_current_board = self.network(current_board_features.unsqueeze(0).to(DEVICE), is_actor_pass=False) # Add batch dim
        self.network.train() # Back to train mode

        action_probs = F.softmax(action_scores, dim=0) # Probabilities over possible S'
        dist = Categorical(action_probs)
        chosen_idx = dist.sample()
        
        log_prob_tensor = dist.log_prob(chosen_idx)
        entropy_tensor = dist.entropy()

        chosen_action_tuple = possible_actions_tuples[chosen_idx.item()]
        features_after_chosen_action = feature_tensors_for_next_steps[chosen_idx.item()]

        aux_info = {
            'log_prob': log_prob_tensor,
            'entropy': entropy_tensor,
            'features_after_chosen_action': features_after_chosen_action, # S' features
            'value_of_current_board': value_of_current_board.squeeze() # V(S_current)
        }
        return chosen_action_tuple, aux_info

    def learn(self, current_board_features, action_tuple, reward, next_board_features_actual, done,
              game_instance_at_s=None, game_instance_at_s_prime=None, aux_info=None):
        """
        A2C learns on-policy from each step.
        `current_board_features`: S_t (board before current piece placement)
        `aux_info['log_prob']`: log_prob of choosing action leading to S_t'
        `aux_info['entropy']`: entropy of policy at S_t
        `aux_info['value_of_current_board']`: V(S_t) estimated by critic
        `reward`: R_{t+1} (reward for placing the piece)
        `next_board_features_actual`: S_{t+1} (board after new piece appears)
        `done`: game over flag for S_{t+1}
        """
        log_prob = aux_info['log_prob']
        entropy = aux_info['entropy']
        value_s_current = aux_info['value_of_current_board'] # This is V(S_t)

        # Convert to tensors
        reward_tensor = torch.tensor([reward], dtype=torch.float32).to(DEVICE)
        done_tensor = torch.tensor([done], dtype=torch.float32).to(DEVICE)
        
        # Calculate V(S_{t+1})
        value_s_next_actual = torch.tensor([0.0], dtype=torch.float32).to(DEVICE) # Default if done
        if not done:
            with torch.no_grad(): # Target value should not have gradient
                value_s_next_actual = self.network(next_board_features_actual.unsqueeze(0).to(DEVICE), is_actor_pass=False).squeeze()
        
        # TD Target for critic: R + gamma * V(S_{t+1})
        td_target = reward_tensor + self.gamma * value_s_next_actual * (1 - done_tensor)
        
        # Advantage: A = td_target - V(S_t)
        advantage = (td_target - value_s_current).detach() # Detach advantage for actor loss

        # Actor Loss (Policy Gradient Loss)
        actor_loss = -(log_prob * advantage)
        
        # Critic Loss (Value Loss) - MSE
        # value_s_current was calculated during select_action. We need to ensure gradients
        # flow for it if it's recalculated or if the network call in select_action was part of graph.
        # For simplicity, let's re-evaluate V(S_current) here for the loss calculation to ensure grad.
        re_evaluated_value_s_current = self.network(current_board_features.unsqueeze(0).to(DEVICE), is_actor_pass=False).squeeze()
        critic_loss = F.mse_loss(re_evaluated_value_s_current, td_target.detach())
        # Alternatively, if value_s_current from aux_info has grad_fn:
        # critic_loss = F.mse_loss(value_s_current, td_target.detach())


        # Total Loss
        total_loss = actor_loss + self.value_loss_coeff * critic_loss - self.entropy_coeff * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
        # Optional: torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

    def save(self, filepath=None):
        path = filepath if filepath else global_config.A2C_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.network.state_dict(), path)
        print(f"A2C Agent saved to {path}")

    def load(self, filepath=None):
        path = filepath if filepath else global_config.A2C_MODEL_PATH
        if os.path.exists(path):
            self.network.load_state_dict(torch.load(path, map_location=DEVICE))
            self.network.eval()
            print(f"A2C Agent loaded from {path}")
        else:
            print(f"Error: A2C model not found at {path}")
            raise FileNotFoundError(f"A2C model not found: {path}")