# tetris_rl_agents/agents/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import os
# import copy # No longer needed here if game_instance_at_s_prime isn't used by it

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.tetris import Tetris

import config as global_config
from .base_agent import BaseAgent

BUFFER_SIZE = global_config.DQN_BUFFER_SIZE
BATCH_SIZE = global_config.DQN_BATCH_SIZE
GAMMA = global_config.DQN_GAMMA
LR = global_config.DQN_LR
UPDATE_EVERY = global_config.DQN_UPDATE_EVERY
DEVICE = global_config.DEVICE

class QNetwork(nn.Module):
    """Network structure, can be shared."""
    def __init__(self, state_size, action_size=1, seed=0,
                 fc1_units=global_config.DQN_FC1_UNITS,
                 fc2_units=global_config.DQN_FC2_UNITS):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
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

class ReplayBuffer: # This ReplayBuffer is ALREADY modified as per your last request
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
            "current_board_features_s_t",
            "features_s_prime_chosen",
            "reward",
            "s_prime_actual_features",
            "done"
        ])
        random.seed(seed)

    def add(self, current_board_features_s_t, features_s_prime_chosen, reward,
            s_prime_actual_features, done):
        e = self.experience(current_board_features_s_t, features_s_prime_chosen, reward,
                            s_prime_actual_features, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        features_s_prime_chosen_batch = torch.stack([e.features_s_prime_chosen for e in experiences if e is not None]).float().to(DEVICE)
        rewards_batch = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        s_prime_actual_features_batch = torch.stack([e.s_prime_actual_features for e in experiences if e is not None]).float().to(DEVICE)
        dones_batch = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)
        return (features_s_prime_chosen_batch, rewards_batch, s_prime_actual_features_batch, dones_batch)

    def __len__(self):
        return len(self.memory)

class DQNAgent(BaseAgent): # This DQNAgent will now behave like OriginalDQNAgent
    def __init__(self, state_size, seed=0):
        super().__init__(state_size)
        self._agent_seed = seed
        random.seed(self._agent_seed)
        torch.manual_seed(self._agent_seed)
        if DEVICE.type == 'cuda':
            torch.cuda.manual_seed_all(self._agent_seed)

        self.qnetwork_local = QNetwork(state_size, seed=self._agent_seed).to(DEVICE)
        # REMOVED: self.qnetwork_target = QNetwork(state_size, seed=self._agent_seed + 1).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.criterion = nn.MSELoss() # Added, as OriginalDQNAgent uses it

        # REMOVED: self.hard_update(self.qnetwork_local, self.qnetwork_target)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed=self._agent_seed)
        
        # REMOVED: self.t_step_learn = 0 # No target network updates
        self.t_step_env = 0 # For UPDATE_EVERY (how often to trigger learning from buffer)

        self.epsilon = global_config.DQN_EPSILON_START
        self.epsilon_min = global_config.DQN_EPSILON_MIN
        # Using the same epsilon decay logic as OriginalDQNAgent for direct comparison
        self.num_epochs_elapsed_for_epsilon = 0 # For OriginalDQNAgent style decay
        # self.epsilon_decay_val = (global_config.DQN_EPSILON_START - global_config.DQN_EPSILON_MIN) / global_config.DQN_EPSILON_DECAY_EPOCHS
        # self.current_epoch_for_decay = 0

        print(f"DQNAgent (modified to be like OriginalDQNAgent) initialized. Epsilon: {self.epsilon:.3f}")


    def select_action(self, current_board_features_s_t, tetris_game_instance, epsilon_override=None):
        # This select_action is already very similar to OriginalDQNAgent's
        current_epsilon = epsilon_override if epsilon_override is not None else self.epsilon
        next_steps_dict = tetris_game_instance.get_next_states()
        if not next_steps_dict:
            chosen_action_tuple = (tetris_game_instance.width // 2, 0)
            features_after_chosen = tetris_game_instance.get_state_properties(tetris_game_instance.board).to(DEVICE)
            return chosen_action_tuple, {'features_s_prime_chosen': features_after_chosen, 'current_board_features_s_t': current_board_features_s_t}

        possible_actions_tuples = list(next_steps_dict.keys())
        s_prime_potential_features_list = [s_prime_feat.to(DEVICE) for s_prime_feat in next_steps_dict.values()]
        
        chosen_idx = -1
        if random.random() < current_epsilon:
            chosen_idx = random.randrange(len(possible_actions_tuples))
        else:
            self.qnetwork_local.eval()
            with torch.no_grad():
                s_prime_features_batch = torch.stack(s_prime_potential_features_list)
                action_values = self.qnetwork_local(s_prime_features_batch)
            self.qnetwork_local.train()
            chosen_idx = torch.argmax(action_values).item()

        chosen_action_tuple = possible_actions_tuples[chosen_idx]
        features_s_prime_chosen = s_prime_potential_features_list[chosen_idx]

        return chosen_action_tuple, {'features_s_prime_chosen': features_s_prime_chosen,
                                     'current_board_features_s_t': current_board_features_s_t}

    def _learn_from_experiences(self, experiences, gamma):
        # `experiences` from ReplayBuffer.sample() is:
        # (features_s_prime_chosen_batch, rewards_batch, s_prime_actual_batch, dones_batch)
        features_s_prime_chosen_batch, rewards_batch, s_prime_actual_batch, dones_batch = experiences

        # Q_expected = Q_local(S'_chosen)
        Q_expected = self.qnetwork_local(features_s_prime_chosen_batch)

        # Target: R + gamma * Q_local(S'_{actual})
        self.qnetwork_local.eval() 
        with torch.no_grad(): 
            next_q_values = self.qnetwork_local(s_prime_actual_batch) 
        self.qnetwork_local.train()
        
        Q_targets = rewards_batch + (gamma * next_q_values * (1 - dones_batch))
        
        loss = self.criterion(Q_expected, Q_targets) # Use self.criterion
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # learn_step is called by learn, it adds to memory and calls _learn_from_experiences
    def learn_step(self, current_board_features_s_t, features_s_prime_chosen, reward, 
                   s_prime_actual_features, done):
        self.memory.add(current_board_features_s_t.cpu(), 
                        features_s_prime_chosen.cpu(), 
                        reward,
                        s_prime_actual_features.cpu(), 
                        done)

        # Epsilon decay like OriginalDQNAgent
        self.num_epochs_elapsed_for_epsilon += 1
        self.epsilon = global_config.DQN_EPSILON_MIN + \
                       (max(global_config.DQN_EPSILON_DECAY_EPOCHS - self.num_epochs_elapsed_for_epsilon, 0) *
                        (global_config.DQN_EPSILON_START - global_config.DQN_EPSILON_MIN) / global_config.DQN_EPSILON_DECAY_EPOCHS)
        
        # Optional debug print for epsilon
        # if self.num_epochs_elapsed_for_epsilon % 20 == 0 :
        #      print(f"[DQNAgent Epsilon] Epoch: {self.num_epochs_elapsed_for_epsilon}, Epsilon: {self.epsilon:.4f}")


        # Original DQN learned if buffer was 1/10th full. UPDATE_EVERY was effectively 1 in that case.
        # Let's make UPDATE_EVERY=1 in config for this behavior.
        # The t_step_env logic here is for compatibility if UPDATE_EVERY > 1
        self.t_step_env = (self.t_step_env + 1) % UPDATE_EVERY
        if self.t_step_env == 0:
            # Original logic: if len(replay_memory) < ORIGINAL_REPLAY_MEMORY_SIZE / 10: continue
            # This is now implicitly handled by BATCH_SIZE check before calling sample()
            if len(self.memory) >= BATCH_SIZE and len(self.memory) >= global_config.DQN_BUFFER_SIZE / 10 : # Ensure buffer has min fill
                experiences = self.memory.sample()
                self._learn_from_experiences(experiences, GAMMA)
                
                # NO TARGET NETWORK UPDATE:
                # self.t_step_learn = (self.t_step_learn + 1) % TARGET_UPDATE_EVERY
                # if self.t_step_learn == 0:
                #     self.hard_update(self.qnetwork_local, self.qnetwork_target)

    def learn(self, state_features, action_tuple, reward, next_state_features, done,
              game_instance_at_s=None, game_instance_at_s_prime=None, aux_info=None):
        s_prime_chosen = aux_info['features_s_prime_chosen']
        s_t_for_buffer = aux_info['current_board_features_s_t']

        self.learn_step(s_t_for_buffer, s_prime_chosen, reward, next_state_features, done)

    # REMOVED: def hard_update(...)

    def reset(self):
        # Epsilon decay is per piece. If you wanted epsilon to reset per game (not typical for this DQN style):
        # self.num_epochs_elapsed_for_epsilon = 0 
        # self.epsilon = global_config.DQN_EPSILON_START
        pass 

    def save(self, filename=None):
        path = filename if filename else global_config.DQN_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.qnetwork_local.state_dict(), path) # Save only local network
        print(f"DQNAgent (as Original) saved to {path}")

    def load(self, filename=None):
        path = filename if filename else global_config.DQN_MODEL_PATH
        if os.path.exists(path):
            self.qnetwork_local.load_state_dict(torch.load(path, map_location=DEVICE))
            self.qnetwork_local.eval()
            # No target network to load or sync
            print(f"DQNAgent (as Original) loaded from {path}")
        else:
            print(f"ERROR: No DQNAgent model found at {path}")
            raise FileNotFoundError(f"DQNAgent model not found: {path}")