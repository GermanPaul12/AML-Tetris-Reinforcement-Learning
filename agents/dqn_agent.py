# tetris_rl_agents/agents/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import os
import copy # For deep copying game states if necessary for replay buffer

# Assuming src.tetris is accessible from here.
# If not, adjust Python path or move Tetris class.
# For a cleaner structure, you might eventually move Tetris class out of src
# or make src a proper package.
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # to access config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src')) # to access tetris
from src.tetris import Tetris # If Tetris class is in src/tetris.py

import config as global_config # Tetris project config
from .base_agent import BaseAgent

BUFFER_SIZE = global_config.DQN_BUFFER_SIZE
BATCH_SIZE = global_config.DQN_BATCH_SIZE
GAMMA = global_config.DQN_GAMMA
LR = global_config.DQN_LR
UPDATE_EVERY = global_config.DQN_UPDATE_EVERY
TARGET_UPDATE_EVERY = global_config.DQN_TARGET_UPDATE_EVERY
DEVICE = global_config.DEVICE

class QNetwork(nn.Module):
    """Neural Network Model for Q-value approximation.
    Matches the original DeepQNetwork from the Tetris project.
    Input: 4 features of a board state (usually after a piece has landed).
    Output: Single Q-value for that state.
    """
    def __init__(self, state_size, action_size=1, seed=0, # action_size is effectively 1
                 fc1_units=global_config.DQN_FC1_UNITS,
                 fc2_units=global_config.DQN_FC2_UNITS):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size) # Output a single Q-value
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

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        # Experience:
        # current_board_features: Features of board before piece placement decision.
        # chosen_action_tuple: The (x, rot) action taken.
        # features_after_chosen_action: Features of board IF chosen_action_tuple was taken (from get_next_states).
        # reward: Reward from env.step().
        # next_board_features_actual: Features of board after env.step() and new piece.
        # done: Game over flag.
        # next_potential_next_state_features: List of feature Tensors for all states reachable from next_board_features_actual.
        self.experience = namedtuple("Experience", field_names=[
            "current_board_features", "features_after_chosen_action", "reward",
            "next_board_features_actual", "done", "next_potential_next_state_features_list"
        ])
        random.seed(seed)

    def add(self, current_board_features, features_after_chosen_action, reward,
            next_board_features_actual, done, next_potential_next_state_features_list):
        e = self.experience(current_board_features, features_after_chosen_action, reward,
                            next_board_features_actual, done, next_potential_next_state_features_list)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        # current_board_features_batch is not directly used by Q-network in this Tetris setup.
        # features_after_chosen_action_batch is used for Q_expected.
        features_after_chosen_action_batch = torch.stack([e.features_after_chosen_action for e in experiences if e is not None]).float().to(DEVICE)
        rewards_batch = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        # next_board_features_actual_batch is used to generate inputs for Q_targets_next.
        # dones_batch is used to zero out Q_target if episode ended.
        dones_batch = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)

        # This is the tricky part: Q_targets_next needs evaluation of multiple states for each item in batch.
        # We pre-store the list of these features.
        next_potential_features_list_batch = [e.next_potential_next_state_features_list for e in experiences if e is not None]

        return (features_after_chosen_action_batch, rewards_batch, dones_batch, next_potential_features_list_batch)

    def __len__(self):
        return len(self.memory)

class DQNAgent(BaseAgent):
    def __init__(self, state_size, seed=0): # action_size not needed for QNetwork here
        super().__init__(state_size)
        self._agent_seed = seed
        random.seed(self._agent_seed)
        torch.manual_seed(self._agent_seed)
        if DEVICE.type == 'cuda':
            torch.cuda.manual_seed_all(self._agent_seed)

        self.qnetwork_local = QNetwork(state_size, seed=self._agent_seed).to(DEVICE)
        self.qnetwork_target = QNetwork(state_size, seed=self._agent_seed).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.hard_update(self.qnetwork_local, self.qnetwork_target)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed=self._agent_seed)
        
        self.t_step_learn = 0 # For TARGET_UPDATE_EVERY (counts learning steps)
        self.t_step_env = 0 # For UPDATE_EVERY (counts environment steps)

        self.epsilon = global_config.DQN_EPSILON_START
        self.epsilon_min = global_config.DQN_EPSILON_MIN
        # Epsilon decay calculated based on num_decay_epochs from original Tetris
        self.epsilon_decay_val = (global_config.DQN_EPSILON_START - global_config.DQN_EPSILON_MIN) / global_config.DQN_EPSILON_DECAY_EPOCHS
        self.current_epoch_for_decay = 0


    def select_action(self, current_board_features, tetris_game_instance, epsilon_override=None):
        """
        Args:
            current_board_features (torch.Tensor): Features of the board before piece placement.
                                                   Not directly used for Q-value calculation here,
                                                   but good for state consistency.
            tetris_game_instance (Tetris): Game object to get next possible states.
        Returns:
            action_tuple (tuple): (x_pos, rotation_idx)
            aux_info (dict): Contains 'features_after_chosen_action'
        """
        current_epsilon = epsilon_override if epsilon_override is not None else self.epsilon

        next_steps_dict = tetris_game_instance.get_next_states()
        possible_actions_tuples = list(next_steps_dict.keys())
        
        # corresponding_next_state_features is a list of Tensors
        corresponding_next_state_features = list(next_steps_dict.values())

        if not possible_actions_tuples: # Should not happen in normal Tetris play
            # Default or error action if no moves are possible (e.g. game almost over)
            # This case needs careful handling based on Tetris game logic
            # For now, let's assume there's always at least one action.
            # If tetris_game_instance.gameover is True, this might be called.
            # Simplest is to pick first, or a "do nothing" if that exists.
            # The original Tetris code implies get_next_states always returns something.
            # If game is over, this select_action shouldn't even be called.
            # Fallback: a default action, e.g., the first one or a predefined one.
            # However, Tetris logic should prevent this. If it happens, it's an issue.
            # print("WARNING: No possible actions in DQN select_action. This is unexpected.")
            # For safety, if this occurs, choose a dummy action (e.g., first if available, or default)
            if not possible_actions_tuples: # Should be rare, implies game might be stuck or over
                 # Fallback: choose the first available action or a default (0,0) if list is empty
                chosen_action_tuple = (tetris_game_instance.width // 2, 0) if not possible_actions_tuples else possible_actions_tuples[0]
                features_after_chosen = tetris_game_instance.get_state_properties(tetris_game_instance.board) # current board state
                return chosen_action_tuple, {'features_after_chosen_action': features_after_chosen.to(DEVICE)}


        if random.random() < current_epsilon: # Exploration
            chosen_idx = random.randrange(len(possible_actions_tuples))
        else: # Exploitation
            self.qnetwork_local.eval()
            with torch.no_grad():
                # Stack features for all possible next states into a batch
                next_state_features_batch = torch.stack(corresponding_next_state_features).to(DEVICE)
                # Get Q-values for all these potential next states
                action_values = self.qnetwork_local(next_state_features_batch)
            self.qnetwork_local.train()
            chosen_idx = torch.argmax(action_values).item()

        chosen_action_tuple = possible_actions_tuples[chosen_idx]
        features_after_chosen_action = corresponding_next_state_features[chosen_idx].to(DEVICE) # Ensure it's on device

        return chosen_action_tuple, {'features_after_chosen_action': features_after_chosen_action}

    def learn_step(self, current_board_features, action_tuple, reward, next_board_features_actual, done,
                   game_instance_at_s=None, game_instance_at_s_prime=None, aux_info=None):
        """
        learn_step is called after env.step().
        `current_board_features`: Features S_t (before action)
        `action_tuple`: Action A_t chosen
        `aux_info['features_after_chosen_action']`: Features S_t' (state if A_t is taken, from get_next_states())
        `reward`: R_{t+1}
        `next_board_features_actual`: Features S_{t+1} (actual state after new piece, from env.step())
        `done`: Game over
        `game_instance_at_s_prime`: Tetris game object in state S_{t+1}
        """
        features_after_chosen_action = aux_info['features_after_chosen_action']

        next_potential_next_state_features_list = []
        if not done and game_instance_at_s_prime:
            # Get all feature vectors for states reachable from S_{t+1}
            # These are S_{t+1}'', S_{t+1}''' etc.
            next_steps_from_s_prime = game_instance_at_s_prime.get_next_states()
            if next_steps_from_s_prime: # Ensure there are moves
                 next_potential_next_state_features_list = [fs.to(DEVICE) for fs in next_steps_from_s_prime.values()]


        self.memory.add(current_board_features.to(DEVICE), features_after_chosen_action, reward,
                        next_board_features_actual.to(DEVICE), done, next_potential_next_state_features_list)

        self.t_step_env = (self.t_step_env + 1) % UPDATE_EVERY
        if self.t_step_env == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self._learn_from_experiences(experiences, GAMMA)
                
                self.t_step_learn = (self.t_step_learn + 1) % TARGET_UPDATE_EVERY
                if self.t_step_learn == 0:
                    self.hard_update(self.qnetwork_local, self.qnetwork_target)
        
        # Epsilon decay is based on "epochs" from original Tetris, which are piece placements/learning steps
        self.current_epoch_for_decay +=1
        if self.current_epoch_for_decay <= global_config.DQN_EPSILON_DECAY_EPOCHS:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay_val)


    def _learn_from_experiences(self, experiences, gamma):
        features_after_chosen_action_batch, rewards_batch, dones_batch, \
            next_potential_features_list_batch = experiences

        # Q_expected = Q_local(state_after_chosen_action)
        Q_expected = self.qnetwork_local(features_after_chosen_action_batch)

        Q_targets_next_max = torch.zeros(BATCH_SIZE, 1, device=DEVICE)
        for i, potential_next_features_list in enumerate(next_potential_features_list_batch):
            if not dones_batch[i].item() and potential_next_features_list: # If not done and there are next moves
                # Stack all potential next-next-state features for this experience item
                next_next_state_features_tensor = torch.stack(potential_next_features_list)
                # Get Q values from target network for all these S''
                q_values_for_next_next_states = self.qnetwork_target(next_next_state_features_tensor).detach()
                Q_targets_next_max[i] = q_values_for_next_next_states.max(0)[0] # Max Q value among S''

        Q_targets = rewards_batch + (gamma * Q_targets_next_max * (1 - dones_batch))
        
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def reset(self): # Called at the start of a new game (not epoch/piece)
        # Epsilon decay happens per learning step (epoch), not per game reset.
        # If there's any game-specific reset logic, add here.
        pass

    def save(self, filename=None):
        path = filename if filename else global_config.DQN_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.qnetwork_local.state_dict(), path)
        print(f"DQN Agent saved to {path}")

    def load(self, filename=None):
        path = filename if filename else global_config.DQN_MODEL_PATH
        if os.path.exists(path):
            self.qnetwork_local.load_state_dict(torch.load(path, map_location=DEVICE))
            self.qnetwork_target.load_state_dict(torch.load(path, map_location=DEVICE))
            self.qnetwork_local.eval()
            self.qnetwork_target.eval()
            print(f"DQN Agent loaded from {path}")
        else:
            print(f"ERROR: No DQN model found at {path}")
            raise FileNotFoundError(f"DQN model not found: {path}")