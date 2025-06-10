# tetris_rl_agents/agents/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import os

from src.tetris import Tetris 
import config as global_config
from .base_agent import BaseAgent

# --- Constants from global_config ---
BUFFER_SIZE = global_config.DQN_BUFFER_SIZE
BATCH_SIZE = global_config.DQN_BATCH_SIZE
GAMMA = global_config.DQN_GAMMA
LR = global_config.DQN_LR

FC1_UNITS = global_config.DQN_FC1_UNITS
FC2_UNITS = global_config.DQN_FC2_UNITS
DEVICE = global_config.DEVICE

# Epsilon parameters (used if no override, or for initial value)
AGENT_EPSILON_START = global_config.DQN_EPSILON_START
AGENT_EPSILON_MIN = global_config.DQN_EPSILON_MIN
# AGENT_EPSILON_DECAY_LEARNING_STEPS = global_config.DQN_EPSILON_DECAY_EPOCHS # No longer primary for agent decay

# --- V-Network Definition (Value Network) --- (No change from previous V-learning version)
class VNetwork(nn.Module):
    def __init__(self, state_size, seed=0, fc1_units=FC1_UNITS, fc2_units=FC2_UNITS):
        super(VNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state_features):
        x = F.relu(self.fc1(state_features))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- Replay Buffer --- (No change from previous V-learning version)
Experience = namedtuple("Experience", field_names=["s_t_features", "s_prime_chosen_features", "reward", "done"])

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        random.seed(seed)
    def add(self, s_t_features, s_prime_chosen_features, reward, done):
        e = Experience(s_t_features, s_prime_chosen_features, reward, done)
        self.memory.append(e)
    def sample(self):
        actual_sample_size = min(len(self.memory), self.batch_size)
        if actual_sample_size == 0: 
            return (torch.empty(0,device=DEVICE),torch.empty(0,device=DEVICE),torch.empty(0,device=DEVICE),torch.empty(0,device=DEVICE))
        experiences = random.sample(self.memory, k=actual_sample_size)
        valid_experiences = [e for e in experiences if e and e.s_t_features is not None and e.s_prime_chosen_features is not None]
        if not valid_experiences: 
            return (torch.empty(0,device=DEVICE),torch.empty(0,device=DEVICE),torch.empty(0,device=DEVICE),torch.empty(0,device=DEVICE))
        s_t_b = torch.stack([e.s_t_features for e in valid_experiences]).float().to(DEVICE)
        s_pc_b = torch.stack([e.s_prime_chosen_features for e in valid_experiences]).float().to(DEVICE)
        r_b = torch.from_numpy(np.vstack([e.reward for e in valid_experiences])).float().to(DEVICE)
        d_b = torch.from_numpy(np.vstack([e.done for e in valid_experiences]).astype(np.uint8)).float().to(DEVICE)
        return (s_t_b, s_pc_b, r_b, d_b)
    def __len__(self): return len(self.memory)


class DQNAgent(BaseAgent):
    def __init__(self, state_size, seed=0):
        super().__init__(state_size)
        self._agent_seed = seed
        random.seed(self._agent_seed)
        torch.manual_seed(self._agent_seed)
        if DEVICE.type == 'cuda': 
            torch.cuda.manual_seed_all(self._agent_seed)

        self.v_network = VNetwork(state_size, seed=self._agent_seed).to(DEVICE)
        self.optimizer = optim.Adam(self.v_network.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed=self._agent_seed)
        
        self.learning_steps_done = 0 # Incremented by train.py after a successful learning update
        self.total_pieces_placed_overall = 0
        
        # self.epsilon is mainly for evaluation if train.py doesn't provide an override
        # For training, train.py calculates and passes epsilon.
        self.epsilon = AGENT_EPSILON_START # Default starting epsilon
        self.last_loss = None 

        print("DQN Agent (V-Learning Style for Original Loop by train.py) initialized.")
        print(f"  Agent's internal epsilon starts at: {self.epsilon:.3f} (train.py will override during training steps)")

    def select_action(self, current_board_features_s_t: torch.Tensor, 
                  tetris_game_instance: Tetris, 
                  epsilon_override: float = None) -> tuple: # train.py will pass epsilon_override
        # Use epsilon_override if provided by train.py, otherwise use agent's internal epsilon
        current_epsilon_for_decision = epsilon_override if epsilon_override is not None else self.epsilon
        
        next_steps_dict = tetris_game_instance.get_next_states()
        if not next_steps_dict:
            chosen_action_tuple = (tetris_game_instance.width // 2, 0) 
            features_s_prime_chosen = current_board_features_s_t 
            return chosen_action_tuple, {
                'features_s_prime_chosen': features_s_prime_chosen.to(DEVICE), 
                'current_board_features_s_t': current_board_features_s_t.to(DEVICE)
            }

        possible_actions_tuples = list(next_steps_dict.keys())
        s_prime_potential_features_list = [s_prime_feat.to(DEVICE) for s_prime_feat in next_steps_dict.values()]
        
        chosen_idx = -1
        if random.random() <= current_epsilon_for_decision: # Use "<=" to match original
            chosen_idx = random.randrange(len(possible_actions_tuples))
        else:
            self.v_network.eval()
            with torch.no_grad():
                s_prime_features_batch = torch.stack(s_prime_potential_features_list)
                v_values_for_s_primes = self.v_network(s_prime_features_batch)
            self.v_network.train()
            chosen_idx = torch.argmax(v_values_for_s_primes).item()

        chosen_action_tuple = possible_actions_tuples[chosen_idx]
        features_s_prime_chosen = s_prime_potential_features_list[chosen_idx]
        
        return chosen_action_tuple, {
            'features_s_prime_chosen': features_s_prime_chosen,
            'current_board_features_s_t': current_board_features_s_t.to(DEVICE)
        }

    def _learn_from_experiences(self, experiences: tuple, gamma: float):
        # (No change from previous V-learning version, this is correct)
        s_t_b, s_prime_chosen_b, rewards_b, dones_b = experiences
        if s_t_b.nelement() == 0: 
            self.last_loss = None 
            return
        v_expected = self.v_network(s_t_b)
        with torch.no_grad(): 
            v_of_s_prime_chosen = self.v_network(s_prime_chosen_b)
        v_targets = rewards_b + (gamma * v_of_s_prime_chosen * (1 - dones_b))
        loss = self.criterion(v_expected, v_targets.detach())
        self.optimizer.zero_grad() 
        loss.backward() 
        self.optimizer.step()
        self.last_loss = loss.item()

    # update_epsilon method is no longer needed in the agent if train.py controls it.
    # The agent's self.epsilon can be updated if necessary for non-training scenarios,
    # but the training epsilon comes from train.py.

    def learn(self, state_features, action_tuple, reward, next_state_features, done,
            game_instance_at_s=None, game_instance_at_s_prime=None, aux_info=None):
        # (No change from previous V-learning version where this just adds to memory)
        if aux_info is None: 
            return
        s_t_features_from_aux = aux_info.get('current_board_features_s_t')
        s_prime_chosen_features_from_aux = aux_info.get('features_s_prime_chosen')
        if s_t_features_from_aux is None or s_prime_chosen_features_from_aux is None: 
            return
        self.memory.add(s_t_features_from_aux.cpu(), 
                        s_prime_chosen_features_from_aux.cpu(), 
                        reward, done)
        self.total_pieces_placed_overall += 1

    def reset(self): # (No change)
        self.last_loss = None 
        pass 

    def save(self, filename=None): # (No change in structure)
        path = filename if filename else global_config.DQN_MODEL_PATH 
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'v_network_state_dict': self.v_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_steps_done': self.learning_steps_done, # Save this important counter
            'total_pieces_placed_overall': self.total_pieces_placed_overall,
            # Save the agent's internal epsilon, though train.py might override it next session
            'epsilon_agent_internal': self.epsilon 
        }, path)
        print(f"DQN Agent (V-Learning, train.py controlled epsilon) saved to {path}")

    def load(self, filename=None): # (Minor adjustment for epsilon loading)
        path = filename if filename else global_config.DQN_MODEL_PATH
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            self.v_network.load_state_dict(checkpoint['v_network_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.learning_steps_done = checkpoint.get('learning_steps_done', 0)
            self.total_pieces_placed_overall = checkpoint.get('total_pieces_placed_overall', 0)
            # Load the agent's internal epsilon. train.py will recalculate based on loaded learning_steps_done.
            self.epsilon = checkpoint.get('epsilon_agent_internal', AGENT_EPSILON_START)
            self.v_network.train()
            print(f"DQN Agent (V-Learning, train.py controlled epsilon) loaded from {path}. "
                  f"Loaded learning steps: {self.learning_steps_done}, Agent internal epsilon: {self.epsilon:.4f}")
        else:
            print(f"ERROR: No DQN V-Learning model (train.py controlled epsilon) found at {path}.")