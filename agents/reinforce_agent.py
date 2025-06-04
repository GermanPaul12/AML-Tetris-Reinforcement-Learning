# tetris_rl_agents/agents/reinforce_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.tetris import Tetris # For type hinting

import config as global_config
from .base_agent import BaseAgent

DEVICE = global_config.DEVICE

class PolicyNetworkREINFORCE(nn.Module):
    """
    Policy Network for REINFORCE.
    Input: 4 features of a board state (after a piece has landed).
    Output: Single logit/score for that state.
    """
    def __init__(self, state_size, action_size=1, seed=0, # action_size is effectively 1
                 fc1_units=global_config.REINFORCE_FC1_UNITS,
                 fc2_units=global_config.REINFORCE_FC2_UNITS):
        super(PolicyNetworkREINFORCE, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size) # Output a single score/logit
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state_features):
        x = F.relu(self.fc1(state_features))
        x = F.relu(self.fc2(x))
        return self.fc3(x) # Raw score/logit

class REINFORCEAgent(BaseAgent):
    def __init__(self, state_size, seed=0,
                 learning_rate=global_config.REINFORCE_LEARNING_RATE,
                 gamma=global_config.REINFORCE_GAMMA):
        super().__init__(state_size)
        self._agent_seed = seed
        self.gamma = gamma
        
        random.seed(self._agent_seed)
        torch.manual_seed(self._agent_seed)
        if DEVICE.type == 'cuda':
            torch.cuda.manual_seed_all(self._agent_seed)

        self.policy_network = PolicyNetworkREINFORCE(state_size, seed=self._agent_seed).to(DEVICE)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.saved_log_probs = [] # Stores log_prob tensors for the chosen actions
        self.rewards = []         # Stores rewards for each piece placement in an episode (game)
        print(f"REINFORCE Agent initialized. LR: {learning_rate}, Gamma: {gamma}")

    def select_action(self, current_board_features: torch.Tensor, tetris_game_instance: Tetris, epsilon_override=None):
        """
        Args:
            current_board_features: Features of board S_t (before piece placement decision). Not directly used by policy net.
            tetris_game_instance: Game object to get next possible states.
        Returns:
            action_tuple (tuple): (x_pos, rotation_idx)
            aux_info (dict): Contains 'log_prob' of the chosen action (distribution over outcomes)
                             and 'features_after_chosen_action'.
        """
        next_steps_dict = tetris_game_instance.get_next_states()
        if not next_steps_dict:
            # This should ideally not happen if the game is playable.
            # If it does, we need a fallback or to ensure game logic prevents it.
            # print("Warning: REINFORCE select_action found no possible moves.")
            # Fallback to a default action, e.g. middle placement, no rotation
            chosen_action_tuple = (tetris_game_instance.width // 2, 0)
            # A dummy log_prob; this state won't contribute much if it leads to game over.
            log_prob_tensor = torch.tensor(0.0, device=DEVICE, requires_grad=True)
            dummy_features = tetris_game_instance.get_state_properties(tetris_game_instance.board).to(DEVICE)
            self.saved_log_probs.append(log_prob_tensor) # Still need to append for list length consistency
            return chosen_action_tuple, {'log_prob': log_prob_tensor, 'features_after_chosen_action': dummy_features}


        possible_actions_tuples = list(next_steps_dict.keys())
        # feature_tensors_for_next_steps are S_t' (features if action taken)
        feature_tensors_for_next_steps = torch.stack(list(next_steps_dict.values())).to(DEVICE)

        self.policy_network.train() # Ensure network is in training mode for gradient tracking
        action_scores = self.policy_network(feature_tensors_for_next_steps).squeeze() # Remove unnecessary dim if any

        if action_scores.dim() == 0: # If only one action possible, squeeze makes it scalar
            action_scores = action_scores.unsqueeze(0)
            
        action_probs = F.softmax(action_scores, dim=0)
        dist = Categorical(action_probs)
        
        # chosen_idx refers to the index in possible_actions_tuples and feature_tensors_for_next_steps
        chosen_idx_tensor = dist.sample() 
        log_prob_tensor = dist.log_prob(chosen_idx_tensor)
        
        self.saved_log_probs.append(log_prob_tensor)

        chosen_action_tuple = possible_actions_tuples[chosen_idx_tensor.item()]
        features_after_chosen = feature_tensors_for_next_steps[chosen_idx_tensor.item()]
        
        return chosen_action_tuple, {'log_prob': log_prob_tensor, 'features_after_chosen_action': features_after_chosen}

    def store_reward(self, reward): # Called by the training loop after env.step()
        self.rewards.append(reward)

    def learn_episode(self): # Called at the end of a full game
        if not self.saved_log_probs: # No actions taken or an issue
            self.reset_episode_data()
            return

        discounted_returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_returns.insert(0, R)
        
        discounted_returns_tensor = torch.tensor(discounted_returns, dtype=torch.float32).to(DEVICE)
        if len(discounted_returns_tensor) > 1: # Normalize if more than one step
            discounted_returns_tensor = (discounted_returns_tensor - discounted_returns_tensor.mean()) / (discounted_returns_tensor.std() + 1e-9)

        policy_loss_terms = []
        if len(self.saved_log_probs) != len(discounted_returns_tensor):
            # This can happen if select_action was called more times than store_reward,
            # e.g. if game ends abruptly after select_action but before store_reward.
            # Or if store_reward was missed. For simplicity, trim to shorter length.
            # print(f"Warning: Mismatch REINFORCE log_probs ({len(self.saved_log_probs)}) and returns ({len(discounted_returns_tensor)}). Trimming.")
            min_len = min(len(self.saved_log_probs), len(discounted_returns_tensor))
            self.saved_log_probs = self.saved_log_probs[:min_len]
            discounted_returns_tensor = discounted_returns_tensor[:min_len]
            if not min_len:
                self.reset_episode_data()
                return


        for log_prob, G_t in zip(self.saved_log_probs, discounted_returns_tensor):
            policy_loss_terms.append(-log_prob * G_t) # G_t does not need .detach() here

        if not policy_loss_terms:
            self.reset_episode_data()
            return

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss_terms).sum()
        
        if policy_loss.requires_grad: # Ensure loss can propagate gradients
            policy_loss.backward()
            self.optimizer.step()
        else:
            # print("Warning (REINFORCE): policy_loss does not require grad. Skipping update.")
            pass # Or log an error

        self.reset_episode_data()

    # The generic learn() in BaseAgent is 'pass'.
    # REINFORCE learns at episode end. The train.py loop will call store_reward() and learn_episode().
    def learn(self, state_features, action_tuple, reward, next_state_features, done, **kwargs):
        self.store_reward(reward) # Store reward for the piece placement
        # If done (game over), the main training loop should call learn_episode()

    def reset_episode_data(self):
        self.saved_log_probs = []
        self.rewards = []

    def reset(self): # Called at the start of each new game by training loop
        self.reset_episode_data()

    def save(self, filename=None):
        path = filename if filename else global_config.REINFORCE_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_network.state_dict(), path)
        print(f"REINFORCE Agent saved to {path}")

    def load(self, filename=None):
        path = filename if filename else global_config.REINFORCE_MODEL_PATH
        if os.path.exists(path):
            self.policy_network.load_state_dict(torch.load(path, map_location=DEVICE))
            self.policy_network.eval()
            print(f"REINFORCE Agent loaded from {path}")
        else:
            print(f"ERROR: No REINFORCE model found at {path}")
            raise FileNotFoundError(f"REINFORCE model not found: {path}")