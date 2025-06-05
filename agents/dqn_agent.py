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

import config as global_config # Make sure this is your project's config
from .base_agent import BaseAgent

BUFFER_SIZE = global_config.DQN_BUFFER_SIZE
BATCH_SIZE = global_config.DQN_BATCH_SIZE
GAMMA = global_config.DQN_GAMMA
LR = global_config.DQN_LR
EPSILON_START = global_config.DQN_EPSILON_START
EPSILON_MIN = global_config.DQN_EPSILON_MIN
EPSILON_DECAY_EPOCHS = global_config.DQN_EPSILON_DECAY_EPOCHS

FC1_UNITS = global_config.DQN_FC1_UNITS
FC2_UNITS = global_config.DQN_FC2_UNITS
DEVICE = global_config.DEVICE

# --- V-Network Definition (Value Network, similar to original DeepQNetwork) ---
class VNetwork(nn.Module): # Renamed for clarity, but structure is the same
    def __init__(self, state_size, seed=0, fc1_units=FC1_UNITS, fc2_units=FC2_UNITS):
        super(VNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Original DeepQNetwork: Linear(4, 64), ReLU, Linear(64, 64), ReLU, Linear(64, 1)
        # state_size is 4 for Tetris features
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1) # Outputs a single V-value for the input state
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state_features): # state_features can be S_t or S'_chosen
        x = F.relu(self.fc1(state_features))
        x = F.relu(self.fc2(x))
        return self.fc3(x) # This is V(state_features)

# --- Replay Buffer (Aligned with original's needs for V-learning) ---
Experience = namedtuple("Experience", field_names=[
    "s_t_features",             # Features of board S_t (where action was chosen)
    "s_prime_chosen_features",  # Features of board S'_chosen (result of chosen action)
    "reward",                   # Reward R_{t+1}
    "done"                      # Done flag (True if S'_chosen leads to terminal state next)
])

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        random.seed(seed) # Python's random for sampling

    # s_t_plus_1_features (from game step) is NOT directly used by this learning formulation
    def add(self, s_t_features, s_prime_chosen_features, reward, done):
        e = Experience(s_t_features, s_prime_chosen_features, reward, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        valid_experiences = [e for e in experiences if e is not None and 
                             e.s_t_features is not None and
                             e.s_prime_chosen_features is not None] # Basic check
        
        if not valid_experiences: # Should not happen if buffer is sufficiently full
             return (torch.empty(0, device=DEVICE), torch.empty(0, device=DEVICE), 
                     torch.empty(0, device=DEVICE), torch.empty(0, device=DEVICE))

        s_t_features_b = torch.stack([e.s_t_features for e in valid_experiences]).float().to(DEVICE)
        s_prime_chosen_features_b = torch.stack([e.s_prime_chosen_features for e in valid_experiences]).float().to(DEVICE)
        rewards_b = torch.from_numpy(np.vstack([e.reward for e in valid_experiences])).float().to(DEVICE)
        dones_b = torch.from_numpy(np.vstack([e.done for e in valid_experiences]).astype(np.uint8)).float().to(DEVICE)
        
        return (s_t_features_b, s_prime_chosen_features_b, rewards_b, dones_b)

    def __len__(self):
        return len(self.memory)

class DQNAgent(BaseAgent): # Technically a "ValueIterationAgent" or "DeepVLearningAgent" now
    def __init__(self, state_size, seed=0):
        super().__init__(state_size)
        self._agent_seed = seed
        random.seed(self._agent_seed) # For agent's own exploration choices
        torch.manual_seed(self._agent_seed)
        if DEVICE.type == 'cuda':
            torch.cuda.manual_seed_all(self._agent_seed)

        self.v_network = VNetwork(state_size, seed=self._agent_seed).to(DEVICE)
        # Target network is not used in this specific V-learning formulation from original train.py
        # self.v_network_target = VNetwork(state_size, seed=self._agent_seed + 1).to(DEVICE) 
        self.optimizer = optim.Adam(self.v_network.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed=self._agent_seed)
        
        self.learning_steps_done = 0 # Counter for epsilon decay, matches original's 'epoch'
        self.total_pieces_placed_overall = 0 # For general tracking if needed, not for epsilon

        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        # Epsilon decay in original was linear over num_decay_epochs (learning steps)
        if EPSILON_DECAY_EPOCHS > 0 :
             self.epsilon_decay_rate_per_step = (EPSILON_START - EPSILON_MIN) / EPSILON_DECAY_EPOCHS
        else:
            self.epsilon_decay_rate_per_step = 0


        self.last_loss = None 

        print(f"DQN Agent (V-Learning Style, adapted from original) initialized. Epsilon: {self.epsilon:.3f}")
        print(f"  State Size: {state_size}, Buffer: {BUFFER_SIZE}, Batch: {BATCH_SIZE}, LR: {LR}")
        print(f"  FC1 Units: {FC1_UNITS}, FC2 Units: {FC2_UNITS}")
        print(f"  Epsilon Decay Learning Steps: {EPSILON_DECAY_EPOCHS}")


    def select_action(self, current_board_features_s_t: torch.Tensor, 
                  tetris_game_instance: Tetris, 
                  epsilon_override: float = None) -> tuple:
        # This method selects action by maximizing V(S'_chosen)
        current_epsilon_val = epsilon_override if epsilon_override is not None else self.epsilon
        next_steps_dict = tetris_game_instance.get_next_states() # {action_tuple: S'_features}

        if not next_steps_dict:
            print("Warning: DQNAgent.select_action called but no next_steps available. Game should be over.")
            # Return a default action; the game outcome will be handled by env.step()
            chosen_action_tuple = (tetris_game_instance.width // 2, 0) 
            #temp_board = [row[:] for row in tetris_game_instance.board] # Make a copy
            features_s_prime_chosen = current_board_features_s_t 
            return chosen_action_tuple, {
                'features_s_prime_chosen': features_s_prime_chosen.to(DEVICE), 
                'current_board_features_s_t': current_board_features_s_t.to(DEVICE)
            }

        possible_actions_tuples = list(next_steps_dict.keys())
        s_prime_potential_features_list = [s_prime_feat.to(DEVICE) for s_prime_feat in next_steps_dict.values()]
        
        chosen_idx = -1
        if random.random() <= current_epsilon_val: # Use "<=" to match original's random() <= epsilon
            chosen_idx = random.randrange(len(possible_actions_tuples))
        else:
            self.v_network.eval() # Set to eval mode for prediction
            with torch.no_grad():
                s_prime_features_batch = torch.stack(s_prime_potential_features_list)
                v_values_for_s_primes = self.v_network(s_prime_features_batch) # V(S')
            self.v_network.train() # Set back to train mode
            chosen_idx = torch.argmax(v_values_for_s_primes).item()

        chosen_action_tuple = possible_actions_tuples[chosen_idx]
        features_s_prime_chosen = s_prime_potential_features_list[chosen_idx] # This is S'_chosen
        
        return chosen_action_tuple, {
            'features_s_prime_chosen': features_s_prime_chosen,
            'current_board_features_s_t': current_board_features_s_t.to(DEVICE) # This is S_t
        }

    def _learn_from_experiences(self, experiences: tuple, gamma: float):
        s_t_b, s_prime_chosen_b, rewards_b, dones_b = experiences

        if s_t_b.nelement() == 0: return # Should not happen if sample is valid

        # V_expected = V_local(S_t)
        v_expected = self.v_network(s_t_b) # Shape: [BATCH_SIZE, 1]

        # Target: R + gamma * V_local(S'_chosen) * (1 - done)
        with torch.no_grad(): # No gradients for target calculation
            v_of_s_prime_chosen = self.v_network(s_prime_chosen_b) # Shape: [BATCH_SIZE, 1]
        
        v_targets = rewards_b + (gamma * v_of_s_prime_chosen * (1 - dones_b))

        loss = self.criterion(v_expected, v_targets.detach()) # Detach targets
        
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: torch.nn.utils.clip_grad_norm_(self.v_network.parameters(), 1.0)
        self.optimizer.step()

        self.last_loss = loss.item()

    def learn(self, state_features, action_tuple, reward, next_state_features, done,
            game_instance_at_s=None, game_instance_at_s_prime=None, aux_info=None):
        
        if aux_info is None:
            print("Error: DQNAgent.learn() expects aux_info. Skipping memory add and learning.")
            return

        s_t_features_from_aux = aux_info.get('current_board_features_s_t')
        s_prime_chosen_features_from_aux = aux_info.get('features_s_prime_chosen')

        if s_t_features_from_aux is None or s_prime_chosen_features_from_aux is None:
            print("Error: DQNAgent.learn() missing critical features in aux_info. Skipping.")
            return
        
        # Store (S_t, S'_chosen, R, Done)
        # `state_features` (S_t from train.py) should be s_t_features_from_aux
        # `next_state_features` (S_{t+1} from train.py) is NOT used in this Bellman update.
        # `done` is the game status after the env.step() that yielded S_{t+1} and `reward`.
        self.memory.add(s_t_features_from_aux.cpu(), 
                        s_prime_chosen_features_from_aux.cpu(), 
                        reward,
                        done)
        
        self.total_pieces_placed_overall += 1 # General counter
        if len(self.memory) >= (BUFFER_SIZE / 10) and len(self.memory) >= BATCH_SIZE:
            experiences = self.memory.sample()
            # Ensure sample is valid (e.g., if buffer just reached batch_size but not BATCH_SIZE/10)
            if experiences[0].size(0) >= self.memory.batch_size : # Or just experiences[0].size(0) > 0
                self._learn_from_experiences(experiences, GAMMA)
                self.learning_steps_done += 1 # Increment learning steps counter

                # Epsilon decay based on learning_steps_done (original's 'epoch')
                if self.epsilon_decay_rate_per_step > 0 and self.learning_steps_done <= EPSILON_DECAY_EPOCHS:
                    self.epsilon = max(EPSILON_MIN, EPSILON_START - self.epsilon_decay_rate_per_step * self.learning_steps_done)
                elif self.learning_steps_done > EPSILON_DECAY_EPOCHS:
                    self.epsilon = EPSILON_MIN
                # If EPSILON_DECAY_EPOCHS is 0, epsilon stays at EPSILON_START or EPSILON_MIN

    def reset(self):
        self.last_loss = None
        # Epsilon is decayed based on learning_steps_done, not reset per game.
        pass 

    def save(self, filename=None):
        path = filename if filename else global_config.DQN_MODEL_PATH # Ensure this path is unique for this agent type
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'v_network_state_dict': self.v_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_steps_done': self.learning_steps_done,
            'total_pieces_placed_overall': self.total_pieces_placed_overall,
            'epsilon': self.epsilon,
        }, path)
        print(f"DQN Agent (V-Learning Style) saved to {path}")

    def load(self, filename=None):
        path = filename if filename else global_config.DQN_MODEL_PATH
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            self.v_network.load_state_dict(checkpoint['v_network_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.learning_steps_done = checkpoint.get('learning_steps_done', 0)
            self.total_pieces_placed_overall = checkpoint.get('total_pieces_placed_overall', 0)
            self.epsilon = checkpoint.get('epsilon', EPSILON_START)
            
            self.v_network.train() # Set to train mode
            print(f"DQN Agent (V-Learning Style) loaded from {path}. Learning steps: {self.learning_steps_done}, Eps: {self.epsilon:.4f}")
        else:
            print(f"ERROR: No DQN V-Learning model found at {path}. Agent starts with initial weights.")