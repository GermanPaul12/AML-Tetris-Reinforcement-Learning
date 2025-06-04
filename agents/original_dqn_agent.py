# tetris_rl_agents/agents/original_dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # Not strictly used in original forward, but good for consistency
import numpy as np
import random
from collections import deque, namedtuple
import os
import copy

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # to access config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src')) # to access tetris
from src.tetris import Tetris

import config as global_config
from .base_agent import BaseAgent

DEVICE = global_config.DEVICE

# --- Parameters from original train.py (can be moved to config.py if desired for this agent) ---
# These are effectively the DQN hyperparameters in config.py, but let's define them
# here to show direct correspondence to the original script's CLI args.
ORIGINAL_REPLAY_MEMORY_SIZE = global_config.DQN_BUFFER_SIZE # opt.replay_memory_size (30000)
ORIGINAL_BATCH_SIZE = global_config.DQN_BATCH_SIZE         # opt.batch_size (512)
ORIGINAL_LR = global_config.DQN_LR                         # opt.lr (1e-3)
ORIGINAL_GAMMA = global_config.DQN_GAMMA                   # opt.gamma (0.99)
ORIGINAL_INITIAL_EPSILON = global_config.DQN_EPSILON_START # opt.initial_epsilon (1)
ORIGINAL_FINAL_EPSILON = global_config.DQN_EPSILON_MIN     # opt.final_epsilon (1e-3)
ORIGINAL_NUM_DECAY_EPOCHS = global_config.DQN_EPSILON_DECAY_EPOCHS # opt.num_decay_epochs (2000)
# Note: The original did not use a separate target network or target_update_every.

class OriginalDeepQNetwork(nn.Module):
    """Replicates the network from src.deep_q_network.py"""
    def __init__(self, state_size=4, fc1_units=global_config.DQN_FC1_UNITS, fc2_units=global_config.DQN_FC2_UNITS, seed=0):
        super(OriginalDeepQNetwork, self).__init__()
        torch.manual_seed(seed) # Seed from BaseAgent will be passed
        # Original used nn.Sequential, functionally similar
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(fc2_units, 1) # Outputs a single Q-value
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x): # x is a batch of state features S'
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x # Batch of Q-values, one for each S'

class OriginalDQNAgent(BaseAgent):
    def __init__(self, state_size, seed=0):
        super().__init__(state_size) # state_size is 4
        self._agent_seed = seed
        random.seed(self._agent_seed)
        torch.manual_seed(self._agent_seed)
        if DEVICE.type == 'cuda':
            torch.cuda.manual_seed_all(self._agent_seed)

        self.model = OriginalDeepQNetwork(state_size=self.state_size, seed=self._agent_seed).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=ORIGINAL_LR)
        self.criterion = nn.MSELoss()

        self.replay_memory = deque(maxlen=ORIGINAL_REPLAY_MEMORY_SIZE)
        self.experience = namedtuple("Experience", field_names=["s_prime_features", "reward", "s_prime_actual_features", "done"])
        
        self.epsilon = ORIGINAL_INITIAL_EPSILON
        self.num_epochs_elapsed = 0 # Tracks piece placements for epsilon decay

        print(f"OriginalDQN Agent initialized. Epsilon starts at {self.epsilon:.3f}")

    def select_action(self, current_board_features_s_t: torch.Tensor, tetris_game_instance: Tetris, epsilon_override=None):
        """
        Selects action based on epsilon-greedy policy over Q-values of potential next states (S').
        `current_board_features_s_t` is not directly used by the model for Q-value prediction in this scheme.
        """
        effective_epsilon = epsilon_override if epsilon_override is not None else self.epsilon

        next_steps_dict = tetris_game_instance.get_next_states()
        if not next_steps_dict: # Should be rare if game is playable
            return (tetris_game_instance.width // 2, 0), {'s_prime_chosen_features': tetris_game_instance.get_state_properties(tetris_game_instance.board).to(DEVICE)}

        possible_actions_tuples = list(next_steps_dict.keys())
        # s_prime_potential_features_list are features of S' (if action taken and piece landed)
        s_prime_potential_features_list = [s_prime_feat.to(DEVICE) for s_prime_feat in next_steps_dict.values()]
        
        chosen_idx = -1
        if random.random() <= effective_epsilon: # Exploration
            chosen_idx = random.randrange(len(possible_actions_tuples))
        else: # Exploitation
            self.model.eval() # Set model to evaluation mode
            with torch.no_grad():
                # Stack all S' features for batch prediction
                s_prime_features_batch = torch.stack(s_prime_potential_features_list)
                predictions = self.model(s_prime_features_batch).squeeze(-1) # Get Q(S') for all S'
            chosen_idx = torch.argmax(predictions).item()
            self.model.train() # Back to training mode

        chosen_action_tuple = possible_actions_tuples[chosen_idx]
        s_prime_chosen_features = s_prime_potential_features_list[chosen_idx] # Features of the S' that was chosen

        # aux_info to pass s_prime_chosen_features for storing in replay buffer
        return chosen_action_tuple, {'s_prime_chosen_features': s_prime_chosen_features}

    def learn(self, state_features, action_tuple, reward, 
              next_state_features, done, # Changed from next_board_features_s_prime_actual
              game_instance_at_s=None, game_instance_at_s_prime=None, aux_info=None):
        """
        Stores experience and performs learning update if enough samples in memory.
        `state_features`: S_t (board before action) - not used by original DQN learn step directly from these args for Q-update.
        `aux_info['s_prime_chosen_features']`: S'_chosen (features of board after chosen piece landed). This is 'state' in replay.
        `reward`: R_{t+1}
        `next_state_features`: S'_{actual} (features of board after chosen piece landed AND new piece appeared). This is 'next_state' in replay.
        `done`: game over flag for S'_{actual}
        """
        # `state_features` (which is S_t) is passed but not directly used by the Q-update logic below,
        # which relies on s_prime_chosen_features from aux_info for the "current state" of the Q-learning update.
        # This is to maintain compatibility with the BaseAgent interface.

        s_prime_chosen_features = aux_info['s_prime_chosen_features'] # This is the "state" for the replay buffer

        # Store: [S'_chosen, R, S'_{actual}, Done]
        # `next_state_features` here corresponds to `s_prime_actual_features` from the training loop
        self.replay_memory.append(self.experience(s_prime_chosen_features.cpu(), reward, next_state_features.cpu(), done))
        
        self.num_epochs_elapsed += 1 # Corresponds to original "epoch" counter
        # Epsilon decay logic from original train.py
        self.epsilon = ORIGINAL_FINAL_EPSILON + \
                       (max(ORIGINAL_NUM_DECAY_EPOCHS - self.num_epochs_elapsed, 0) *
                        (ORIGINAL_INITIAL_EPSILON - ORIGINAL_FINAL_EPSILON) / ORIGINAL_NUM_DECAY_EPOCHS)

        if len(self.replay_memory) < ORIGINAL_REPLAY_MEMORY_SIZE / 10:
            return
        if len(self.replay_memory) < ORIGINAL_BATCH_SIZE : 
            return

        batch = random.sample(self.replay_memory, ORIGINAL_BATCH_SIZE)
        s_prime_chosen_batch, reward_batch_np, s_prime_actual_batch, done_batch_np = zip(*batch)

        s_prime_chosen_batch_t = torch.stack(s_prime_chosen_batch).to(DEVICE)
        reward_batch_t = torch.from_numpy(np.array(reward_batch_np, dtype=np.float32)[:, None]).to(DEVICE)
        s_prime_actual_batch_t = torch.stack(s_prime_actual_batch).to(DEVICE)
        done_batch_t = torch.from_numpy(np.array(done_batch_np).astype(np.uint8)[:, None]).to(DEVICE)

        q_values_for_s_prime_chosen = self.model(s_prime_chosen_batch_t)

        self.model.eval() 
        with torch.no_grad(): 
            next_q_values_for_s_prime_actual = self.model(s_prime_actual_batch_t)
        self.model.train()

        y_batch = reward_batch_t + ORIGINAL_GAMMA * next_q_values_for_s_prime_actual * (1 - done_batch_t)
        
        self.optimizer.zero_grad()
        loss = self.criterion(q_values_for_s_prime_chosen, y_batch)
        loss.backward()
        self.optimizer.step()

    def reset(self): # Called at start of a new game by training loop
        # Epsilon decays per piece placement (self.num_epochs_elapsed), not per game.
        # No other game-specific state to reset in this agent.
        pass

    def save(self, filepath=None):
        # The original Tetris saved the model as "epoch_X" or just "tetris".
        # We'll use the config path.
        path = filepath if filepath else global_config.DQN_MODEL_PATH # Could make an ORIGINAL_DQN_MODEL_PATH
        if path == global_config.DQN_MODEL_PATH: # If using the standard DQN path, add a suffix
            base, ext = os.path.splitext(path)
            path = f"{base}_original{ext}"
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Original saved the whole model, not just state_dict.
        # torch.save(self.model, path)
        # It's generally better practice to save state_dict for flexibility:
        torch.save(self.model.state_dict(), path)
        print(f"OriginalDQN Agent model state_dict saved to {path}")

    def load(self, filepath=None):
        path = filepath if filepath else global_config.DQN_MODEL_PATH
        if path == global_config.DQN_MODEL_PATH:
            base, ext = os.path.splitext(path)
            path = f"{base}_original{ext}"

        if os.path.exists(path):
            # If model was saved directly:
            # self.model = torch.load(path, map_location=DEVICE)
            # If state_dict was saved:
            self.model.load_state_dict(torch.load(path, map_location=DEVICE))
            self.model.eval() # Set to evaluation mode
            print(f"OriginalDQN Agent model loaded from {path}")
        else:
            print(f"ERROR: No OriginalDQN model found at {path}")
            raise FileNotFoundError(f"OriginalDQN model not found: {path}")