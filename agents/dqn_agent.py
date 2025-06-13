import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from collections import deque, namedtuple

from helper import *
from src.tetris import Tetris 
import config as global_config
from .base_agent import BaseAgent, PolicyNetwork

DEVICE = global_config.DEVICE

Experience = namedtuple("Experience", field_names=["s_t_features", "s_prime_chosen_features", "reward", "done"])

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, s_t_features, s_prime_chosen_features, reward, done):
        e = Experience(s_t_features, s_prime_chosen_features, reward, done)
        self.memory.append(e)

    def sample(self):
        
        actual_sample_size = min(len(self.memory), self.batch_size)
        if actual_sample_size == 0: return (torch.empty(0,device=DEVICE), torch.empty(0,device=DEVICE), torch.empty(0,device=DEVICE), torch.empty(0,device=DEVICE))
        
        experiences = random.sample(self.memory, k=actual_sample_size)
        valid_experiences = [e for e in experiences if e and e.s_t_features is not None and e.s_prime_chosen_features is not None]
        if not valid_experiences: return (torch.empty(0,device=DEVICE),torch.empty(0,device=DEVICE),torch.empty(0,device=DEVICE),torch.empty(0,device=DEVICE))
        
        s_t_b = torch.stack([e.s_t_features for e in valid_experiences]).float().to(DEVICE)
        s_pc_b = torch.stack([e.s_prime_chosen_features for e in valid_experiences]).float().to(DEVICE)
        r_b = torch.from_numpy(np.vstack([e.reward for e in valid_experiences])).float().to(DEVICE)
        d_b = torch.from_numpy(np.vstack([e.done for e in valid_experiences]).astype(np.uint8)).float().to(DEVICE)
        
        return (s_t_b, s_pc_b, r_b, d_b)
    
    def __len__(self): return len(self.memory)

class DQNAgent(BaseAgent):
    def __init__(self, state_size):
        super().__init__(state_size)

        self.v_network = PolicyNetwork(state_size, fc1_units=global_config.DQN_FC1_UNITS, fc2_units=global_config.DQN_FC2_UNITS).to(DEVICE)
        self.optimizer = optim.Adam(self.v_network.parameters(), lr=global_config.DQN_LR)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(global_config.DQN_BUFFER_SIZE, global_config.DQN_BATCH_SIZE)
        
        self.learning_steps_done = 0
        self.total_pieces_placed_overall = 0

        self.epsilon = global_config.DQN_EPSILON_START
        self.last_loss = None 

        print("DQN Agent (V-Learning Style for Original Loop by train.py) initialized.")
        print(f"  Agent's internal epsilon starts at: {self.epsilon:.3f} (train.py will override during training steps)")

    def select_action(self, current_board_features_s_t: torch.Tensor, tetris_game_instance: Tetris, epsilon_override: float = None) -> tuple: 
        current_epsilon_for_decision = epsilon_override if epsilon_override else self.epsilon
        
        # Get the current board features
        next_steps_dict = tetris_game_instance.get_next_states()
        possible_actions_tuples = list(next_steps_dict.keys())
        possible_features = [features.to(DEVICE) for features in next_steps_dict.values()]
        
        if random.random() <= current_epsilon_for_decision: # Select index randomly
            chosen_idx = random.randrange(len(possible_actions_tuples)) 
        else:
            self.v_network.eval()
            with torch.no_grad(): q_values = self.v_network(torch.stack(possible_features))
            self.v_network.train()
            chosen_idx = torch.argmax(q_values).item()

        chosen_action_tuple = possible_actions_tuples[chosen_idx]
        features_chosen = possible_features[chosen_idx]
        
        return chosen_action_tuple, {
            'features_s_prime_chosen': features_chosen,
            'current_board_features_s_t': current_board_features_s_t.to(DEVICE)
        }

    def learn_from_ReplayBuffer(self, experiences: tuple, gamma: float):
        
        s_t_b, s_prime_chosen_b, rewards_b, dones_b = experiences
        
        if s_t_b.nelement() == 0: self.last_loss = None; return
        
        v_expected = self.v_network(s_t_b)
        with torch.no_grad(): v_of_s_prime_chosen = self.v_network(s_prime_chosen_b)
        
        # Calculate the expected value for the next state
        v_targets = rewards_b + (gamma * v_of_s_prime_chosen * (1 - dones_b))
        loss = self.criterion(v_expected, v_targets.detach())
        
        # Update the learning steps done
        self.optimizer.zero_grad() 
        loss.backward() 
        self.optimizer.step()
        self.last_loss = loss.item()

    def expand_memory(self, reward, done, state_info=None):
        
        if state_info is None: print("Warning: DQNAgent.learn() called without state_info. No memory update will occur."); return
        
        board_features = state_info.get('current_board_features_s_t')
        chosen_features = state_info.get('features_s_prime_chosen')
        
        if board_features is None or chosen_features is None: print("Warning: DQNAgent.learn() called without valid features in aux_info. No memory update will occur."); return
        
        self.memory.add(board_features.cpu(), chosen_features.cpu(), reward, done)
        self.total_pieces_placed_overall += 1

    def reset(self):
        self.last_loss = None 

    def save(self, filename=None):
        path = filename if filename else global_config.DQN_MODEL_PATH 
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'v_network_state_dict': self.v_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_steps_done': self.learning_steps_done,
            'total_pieces_placed_overall': self.total_pieces_placed_overall,
            'epsilon_agent_internal': self.epsilon 
        }, path)
        print(f"DQN Agent (V-Learning, train.py controlled epsilon) saved to {path}")

    def load(self, filename=None):
        path = filename if filename else global_config.DQN_MODEL_PATH
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            self.v_network.load_state_dict(checkpoint['v_network_state_dict'])
            if 'optimizer_state_dict' in checkpoint: self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.learning_steps_done = checkpoint.get('learning_steps_done', 0)
            self.total_pieces_placed_overall = checkpoint.get('total_pieces_placed_overall', 0)
            self.epsilon = checkpoint.get('epsilon_agent_internal', global_config.DQN_EPSILON_START)
            self.v_network.train()
            print(f"DQN Agent (V-Learning, train.py controlled epsilon) loaded from {path}. "
                  f"Loaded learning steps: {self.learning_steps_done}, Agent internal epsilon: {self.epsilon:.4f}")
        else:
            print(f"ERROR: No DQN V-Learning model (train.py controlled epsilon) found at {path}.")
