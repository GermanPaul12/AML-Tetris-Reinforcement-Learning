# tetris_rl_agents/agents/es_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class PolicyNetworkES(nn.Module): # Can be same as GA's PolicyNetwork
    def __init__(self, state_size, action_size=1, seed=0,
                 fc1_units=global_config.ES_FC1_UNITS,
                 fc2_units=global_config.ES_FC2_UNITS):
        super(PolicyNetworkES, self).__init__()
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

    def get_weights_flat(self):
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.parameters()])

    def set_weights_flat(self, flat_weights):
        offset = 0
        for param in self.parameters():
            shape = param.data.shape
            num_elements = param.data.numel()
            param.data.copy_(torch.from_numpy(flat_weights[offset:offset+num_elements]).view(shape).to(DEVICE))
            offset += num_elements
        if offset != len(flat_weights):
            raise ValueError("Size mismatch in set_weights_flat for ES")

class ESAgent(BaseAgent): # ESAgent also acts as the controller
    def __init__(self, state_size, seed=0,
                 population_size=global_config.ES_POPULATION_SIZE,
                 sigma=global_config.ES_SIGMA,
                 learning_rate=global_config.ES_LEARNING_RATE,
                 eval_games_per_param=global_config.ES_EVAL_GAMES_PER_PARAM,
                 max_pieces_per_eval_game=global_config.ES_MAX_PIECES_PER_ES_EVAL_GAME):
        super().__init__(state_size)
        self._agent_seed = seed
        random.seed(self._agent_seed)
        np.random.seed(self._agent_seed) # Crucial for noise generation
        torch.manual_seed(self._agent_seed)

        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.eval_games_per_param = eval_games_per_param
        self.max_pieces_per_eval_game = max_pieces_per_eval_game

        # Central policy network whose parameters are evolved
        self.central_policy_net = PolicyNetworkES(state_size, action_size=1, seed=self._agent_seed).to(DEVICE)
        self.num_params = self.central_policy_net.get_weights_flat().size
        
        self.current_best_fitness = -float('inf')
        self.current_best_weights = self.central_policy_net.get_weights_flat().copy()

        print(f"ES Agent (Controller) initialized. Pop_Size: {self.population_size}, Sigma: {self.sigma}, LR: {self.learning_rate}")

    def _evaluate_parameters(self, flat_weights_candidate, eval_env_template: Tetris):
        temp_policy_net = PolicyNetworkES(self.state_size, action_size=1, seed=0).to(DEVICE) # seed for structure only
        temp_policy_net.set_weights_flat(flat_weights_candidate)
        temp_policy_net.eval()

        total_rewards = []
        for _ in range(self.eval_games_per_param):
            eval_env = Tetris(width=eval_env_template.width,
                              height=eval_env_template.height,
                              block_size=eval_env_template.block_size)
            current_board_features = eval_env.reset()
            if DEVICE.type == 'cuda': current_board_features = current_board_features.cuda()
            
            game_score = 0
            game_over = False
            pieces_this_game = 0

            while not game_over and pieces_this_game < self.max_pieces_per_eval_game:
                next_steps_dict = eval_env.get_next_states()
                if not next_steps_dict: break

                possible_actions_tuples = list(next_steps_dict.keys())
                feature_tensors_for_next_steps = torch.stack(list(next_steps_dict.values())).to(DEVICE)

                with torch.no_grad():
                    action_scores = temp_policy_net(feature_tensors_for_next_steps)
                
                chosen_idx = torch.argmax(action_scores.squeeze()).item()
                chosen_action_tuple = possible_actions_tuples[chosen_idx]

                reward, game_over = eval_env.step(chosen_action_tuple, render=False)
                game_score += reward
                pieces_this_game +=1
                
                if not game_over:
                    current_board_features = eval_env.get_state_properties(eval_env.board)
                    if DEVICE.type == 'cuda': current_board_features = current_board_features.cuda()
            total_rewards.append(game_score)
        return np.mean(total_rewards) if total_rewards else -float('inf')

    def evolve_step(self, eval_env_template: Tetris):
        current_central_weights = self.central_policy_net.get_weights_flat()
        noise_samples = np.random.randn(self.population_size, self.num_params)
        
        fitness_scores = np.zeros(self.population_size)
        perturbed_weights_list = [] # Store weights to evaluate central policy with current noise

        for i in range(self.population_size):
            perturbed_weights = current_central_weights + self.sigma * noise_samples[i]
            perturbed_weights_list.append(perturbed_weights) # Keep for potential re-use if needed
            fitness_scores[i] = self._evaluate_parameters(perturbed_weights, eval_env_template)
        
        # Standardize fitness scores
        if np.std(fitness_scores) > 1e-6:
            standardized_fitness = (fitness_scores - np.mean(fitness_scores)) / np.std(fitness_scores)
        else:
            standardized_fitness = np.zeros_like(fitness_scores)

        update_direction = np.dot(noise_samples.T, standardized_fitness)
        new_central_weights = current_central_weights + (self.learning_rate / (self.population_size * self.sigma)) * update_direction
        self.central_policy_net.set_weights_flat(new_central_weights)

        # Evaluate the new central policy to track its own fitness (and update best_overall)
        current_eval_of_central = self._evaluate_parameters(new_central_weights, eval_env_template)
        if current_eval_of_central > self.current_best_fitness:
            self.current_best_fitness = current_eval_of_central
            self.current_best_weights = new_central_weights.copy()
            print(f"    ES: New best central policy fitness: {self.current_best_fitness:.2f}")
        
        return np.mean(fitness_scores), np.max(fitness_scores), current_eval_of_central

    # select_action uses the current_best_weights
    def select_action(self, current_board_features: torch.Tensor, tetris_game_instance: Tetris, epsilon_override=None):
        # Ensure the central net has the best weights for action selection
        self.central_policy_net.set_weights_flat(self.current_best_weights)
        self.central_policy_net.eval()

        next_steps_dict = tetris_game_instance.get_next_states()
        if not next_steps_dict:
            return (tetris_game_instance.width // 2, 0), {}

        possible_actions_tuples = list(next_steps_dict.keys())
        feature_tensors_for_next_steps = torch.stack(list(next_steps_dict.values())).to(DEVICE)

        with torch.no_grad():
            action_scores = self.central_policy_net(feature_tensors_for_next_steps)
        
        chosen_idx = torch.argmax(action_scores.squeeze()).item()
        chosen_action_tuple = possible_actions_tuples[chosen_idx]
        
        return chosen_action_tuple, {'features_after_chosen_action': feature_tensors_for_next_steps[chosen_idx]}

    def learn(self, *args, **kwargs): # Learning happens in evolve_step
        pass

    def save(self, filepath=None):
        path = filepath if filepath else global_config.ES_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save the best weights by loading them into the network first
        self.central_policy_net.set_weights_flat(self.current_best_weights)
        torch.save(self.central_policy_net.state_dict(), path)
        print(f"ES Agent (best policy) saved to: {path}")

    def load(self, filepath=None):
        path = filepath if filepath else global_config.ES_MODEL_PATH
        if os.path.exists(path):
            self.central_policy_net.load_state_dict(torch.load(path, map_location=DEVICE))
            self.central_policy_net.eval()
            self.current_best_weights = self.central_policy_net.get_weights_flat().copy()
            # self.current_best_fitness would need re-evaluation
            print(f"ES Agent (policy) loaded from {path}. Fitness needs re-evaluation.")
        else:
            print(f"Error: ES model file not found at {path}")
            raise FileNotFoundError(f"ES model not found: {path}")