# tetris_rl_agents/agents/ppo_agent.py
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

class ActorPPO(nn.Module):
    def __init__(self, state_size, seed=0,
                 fc1_units=global_config.PPO_ACTOR_FC1,
                 fc2_units=global_config.PPO_ACTOR_FC2):
        super(ActorPPO, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # This actor head scores a *single potential next state S'*.
        # The policy distribution is formed externally over the scores of all available S'.
        self.fc_actor_head = nn.Linear(fc2_units, 1) 
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.constant_(m.bias, 0)

    def forward(self, state_features_of_potential_next_state_batch):
        # Input can be a batch of S' features [batch_size, num_potential_actions, state_size]
        # or a single S' feature [state_size] or a list of S' features [num_potential_actions, state_size]
        # This network processes one S' feature at a time to output one score.
        # If input is [N, state_size], output is [N, 1]
        x = F.relu(self.fc1(state_features_of_potential_next_state_batch))
        x = F.relu(self.fc2(x))
        return self.fc_actor_head(x)

class CriticPPO(nn.Module):
    def __init__(self, state_size, seed=0,
                 fc1_units=global_config.PPO_CRITIC_FC1,
                 fc2_units=global_config.PPO_CRITIC_FC2):
        super(CriticPPO, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc_critic = nn.Linear(fc2_units, 1) # Outputs value V(S_t)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.constant_(m.bias, 0)

    def forward(self, current_board_state_features):
        x = F.relu(self.fc1(current_board_state_features))
        x = F.relu(self.fc2(x))
        return self.fc_critic(x)

class PPOAgent(BaseAgent):
    def __init__(self, state_size, seed=0,
                 actor_lr=global_config.PPO_ACTOR_LR, critic_lr=global_config.PPO_CRITIC_LR,
                 gamma=global_config.PPO_GAMMA, ppo_epochs=global_config.PPO_EPOCHS_PER_UPDATE,
                 ppo_clip=global_config.PPO_CLIP_EPSILON, batch_size=global_config.PPO_BATCH_SIZE,
                 gae_lambda=global_config.PPO_GAE_LAMBDA, entropy_coeff=global_config.PPO_ENTROPY_COEFF,
                 value_loss_coeff=global_config.PPO_VALUE_LOSS_COEFF,
                 update_horizon=global_config.PPO_UPDATE_HORIZON):
        super().__init__(state_size)
        self._agent_seed = seed
        random.seed(self._agent_seed); torch.manual_seed(self._agent_seed); np.random.seed(self._agent_seed)
        if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(self._agent_seed)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.ppo_clip = ppo_clip
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.batch_size = batch_size
        self.update_horizon = update_horizon

        self.actor = ActorPPO(state_size, seed=self._agent_seed).to(DEVICE)
        self.critic = CriticPPO(state_size, seed=self._agent_seed).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Memory for one update cycle (horizon)
        self.memory_s_t_features = []               # S_t (current board features before action)
        self.memory_all_s_prime_features_lists = [] # List of all S'' features available from S_t
        self.memory_chosen_action_indices = []      # Index of the chosen S' within the list above
        self.memory_log_probs_old_policy = []       # log pi_old(chosen_action_index | S_t)
        self.memory_rewards = []                    # R_{t+1}
        self.memory_dones = []                      # Done flag for S_{t+1}
        self.memory_values_s_t = []                 # V(S_t) from critic

        print(f"PPO Agent initialized. Update Horizon: {self.update_horizon}, Device: {DEVICE}")

    def select_action(self, current_board_features_s_t: torch.Tensor, tetris_game_instance: Tetris,
                      store_in_memory=True, epsilon_override=None):
        next_steps_dict = tetris_game_instance.get_next_states()
        if not next_steps_dict: # Should be rare if game is ongoing
            # Fallback action if no moves are possible (e.g., game almost over or stuck)
            # This indicates an issue or the end of the game, so this selection shouldn't heavily influence learning.
            chosen_action_tuple = (tetris_game_instance.width // 2, 0) # Default middle placement
            # Dummy aux_info. The trainer should ideally not call learn() if game_over was true before this.
            return chosen_action_tuple, {
                'chosen_action_index': 0, # Dummy index
                'log_prob': torch.tensor(0.0, device=DEVICE), # Dummy log_prob
                'entropy': torch.tensor(0.0, device=DEVICE),  # Dummy entropy
                'features_after_chosen_action': tetris_game_instance.get_state_properties(tetris_game_instance.board).to(DEVICE), # Current board state
                'all_available_s_prime_features': [tetris_game_instance.get_state_properties(tetris_game_instance.board).to(DEVICE)] # Dummy list
            }


        possible_actions_tuples = list(next_steps_dict.keys())
        # all_s_prime_features is a list of Tensors, each representing a potential S'
        all_s_prime_features_list = [s_prime_feat.to(DEVICE) for s_prime_feat in next_steps_dict.values()]
        # Stack them for batch processing by the actor
        all_s_prime_features_batch = torch.stack(all_s_prime_features_list)

        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            # Actor scores for all S'
            action_scores = self.actor(all_s_prime_features_batch).squeeze(-1) # Output: [num_potential_actions]
            # Critic value V(S_t)
            value_s_t = self.critic(current_board_features_s_t.unsqueeze(0).to(DEVICE)) # Add batch dim for critic
        self.actor.train()
        self.critic.train()

        action_probs = F.softmax(action_scores, dim=0) # Softmax over scores of S'
        dist = Categorical(action_probs)
        chosen_action_index_tensor = dist.sample() # Samples an index
        chosen_action_index = chosen_action_index_tensor.item()
        
        log_prob_tensor = dist.log_prob(chosen_action_index_tensor) # log pi(chosen_index | S_t)
        entropy_tensor = dist.entropy()

        chosen_action_tuple = possible_actions_tuples[chosen_action_index]
        features_s_prime_chosen = all_s_prime_features_list[chosen_action_index] # The S' that was chosen

        if store_in_memory:
            self.memory_s_t_features.append(current_board_features_s_t.cpu())
            self.memory_all_s_prime_features_lists.append([f.cpu() for f in all_s_prime_features_list]) # Store list of S' features
            self.memory_chosen_action_indices.append(chosen_action_index)
            self.memory_log_probs_old_policy.append(log_prob_tensor.cpu())
            self.memory_values_s_t.append(value_s_t.cpu())

        aux_info = {
            'chosen_action_index': chosen_action_index, # Integer index
            'log_prob': log_prob_tensor,                # Scalar tensor
            'entropy': entropy_tensor,                  # Scalar tensor
            'features_after_chosen_action': features_s_prime_chosen, # Tensor S'_chosen
            'all_available_s_prime_features': all_s_prime_features_list # List of Tensors
        }
        return chosen_action_tuple, aux_info

    def store_transition_result(self, reward, done):
        if len(self.memory_rewards) < len(self.memory_chosen_action_indices):
            self.memory_rewards.append(reward)
            self.memory_dones.append(done)

    def _calculate_advantages_gae(self, last_s_t_plus_1_features_actual: torch.Tensor = None):
        advantages = []
        gae = 0.0
        
        rewards_t = torch.tensor(self.memory_rewards, dtype=torch.float32).to(DEVICE)
        dones_t = torch.tensor(self.memory_dones, dtype=torch.float32).to(DEVICE)
        values_s_t_memory = torch.stack(self.memory_values_s_t).squeeze().to(DEVICE) # V(S_t) from memory

        value_at_horizon_end = torch.tensor([0.0], device=DEVICE) # V(S_T_horizon_actual)
        if last_s_t_plus_1_features_actual is not None and not self.memory_dones[-1]: # If horizon ended mid-game
            with torch.no_grad():
                 value_at_horizon_end = self.critic(last_s_t_plus_1_features_actual.unsqueeze(0).to(DEVICE)).squeeze()
        
        for i in reversed(range(len(rewards_t))):
            # v_s_t_plus_1_actual is V(S_{t+1}_actual)
            # S_{t+1}_actual is the state for which values_s_t_memory[i+1] was V(S_t)
            v_s_t_plus_1_actual = values_s_t_memory[i+1] if i + 1 < len(values_s_t_memory) else value_at_horizon_end
            
            delta = rewards_t[i] + self.gamma * v_s_t_plus_1_actual * (1 - dones_t[i]) - values_s_t_memory[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones_t[i]) * gae
            advantages.insert(0, gae)
        
        advantages_tensor = torch.stack(advantages)
        returns_for_critic_tensor = advantages_tensor + values_s_t_memory # Targets for V(S_t)
        
        if len(advantages_tensor) > 1: # Normalize advantages
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        return advantages_tensor, returns_for_critic_tensor

    def learn_from_memory(self, last_s_t_plus_1_features_actual: torch.Tensor = None):
        if len(self.memory_chosen_action_indices) < self.update_horizon: return

        advantages, returns_for_critic = self._calculate_advantages_gae(last_s_t_plus_1_features_actual)
        
        # Convert memory to tensors for batching
        # S_t (board before current piece placement decision)
        old_s_t_features_b = torch.stack(self.memory_s_t_features).to(DEVICE)
        # List of lists of S' features (all S'' available from each S_t)
        old_all_s_prime_features_lists_b_cpu = self.memory_all_s_prime_features_lists
        
        old_chosen_action_indices_b = torch.tensor(self.memory_chosen_action_indices, dtype=torch.long).to(DEVICE)
        old_log_probs_b = torch.stack(self.memory_log_probs_old_policy).to(DEVICE)

        num_samples = len(self.memory_chosen_action_indices)
        sample_indices = np.arange(num_samples)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(sample_indices)
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx_list = sample_indices[start:end]

                # Minibatch data
                mb_s_t_features = old_s_t_features_b[batch_idx_list]
                # mb_all_s_prime_features_lists contains, for each S_t in batch, a list of S'' Tensors
                mb_all_s_prime_features_lists_cpu = [old_all_s_prime_features_lists_b_cpu[i] for i in batch_idx_list]
                
                mb_chosen_action_indices = old_chosen_action_indices_b[batch_idx_list]
                mb_old_log_probs = old_log_probs_b[batch_idx_list]
                mb_advantages = advantages[batch_idx_list]
                mb_returns_for_critic = returns_for_critic[batch_idx_list] # Targets for V(S_t)

                # --- Actor Loss ---
                new_log_probs_list = []
                entropy_list = []

                # Iterate through the minibatch to handle variable number of next states
                for k_in_batch in range(len(mb_all_s_prime_features_lists_cpu)):
                    s_prime_features_for_this_s_t_cpu = mb_all_s_prime_features_lists_cpu[k_in_batch]
                    if not s_prime_features_for_this_s_t_cpu : # Should not happen if game was playable
                        # Handle empty list: assign dummy log_prob and entropy
                        new_log_probs_list.append(torch.tensor(0.0, device=DEVICE)) # Can't compute log_prob if no actions
                        entropy_list.append(torch.tensor(0.0, device=DEVICE))
                        continue

                    s_prime_features_for_this_s_t_gpu = torch.stack(s_prime_features_for_this_s_t_cpu).to(DEVICE)
                    
                    # Get scores for all S'' from S_t using current actor
                    current_action_scores = self.actor(s_prime_features_for_this_s_t_gpu).squeeze(-1)
                    current_action_probs = F.softmax(current_action_scores, dim=0)
                    current_dist = Categorical(current_action_probs)
                    
                    # Get log_prob of the *originally chosen action index* under the *current policy*
                    log_prob_of_chosen_action_new_policy = current_dist.log_prob(mb_chosen_action_indices[k_in_batch])
                    new_log_probs_list.append(log_prob_of_chosen_action_new_policy)
                    entropy_list.append(current_dist.entropy())

                if not new_log_probs_list: continue # Skip batch if all items had no next states

                new_log_probs_batch = torch.stack(new_log_probs_list)
                entropy_batch = torch.stack(entropy_list).mean() # Mean entropy for the batch

                ratio = torch.exp(new_log_probs_batch - mb_old_log_probs) # Element-wise
                
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy_batch
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                # --- Critic Loss (Value Loss for V(S_t)) ---
                current_values_s_t_new = self.critic(mb_s_t_features).squeeze() # V_new(S_t)
                critic_loss = F.mse_loss(current_values_s_t_new, mb_returns_for_critic)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
        
        self.clear_memory()

    def learn(self, current_board_features, action_tuple, reward, next_board_features_actual, done,
              game_instance_at_s=None, game_instance_at_s_prime=None, aux_info=None):
        # select_action stored: S_t, all_S'_list, chosen_action_idx, log_prob_old, V(S_t)
        # Here, we just store R_{t+1} and D_{t+1}
        self.store_transition_result(reward, done)
        # Training loop in train.py calls learn_from_memory() when horizon is full.

    def clear_memory(self):
        self.memory_s_t_features = []
        self.memory_all_s_prime_features_lists = []
        self.memory_chosen_action_indices = []
        self.memory_log_probs_old_policy = []
        self.memory_rewards = []
        self.memory_dones = []
        self.memory_values_s_t = []

    def save(self, filename_actor=None, filename_critic=None):
        path_actor = filename_actor if filename_actor else global_config.PPO_ACTOR_MODEL_PATH
        path_critic = filename_critic if filename_critic else global_config.PPO_CRITIC_MODEL_PATH
        os.makedirs(os.path.dirname(path_actor), exist_ok=True)
        os.makedirs(os.path.dirname(path_critic), exist_ok=True)
        torch.save(self.actor.state_dict(), path_actor)
        torch.save(self.critic.state_dict(), path_critic)
        print(f"PPO Actor saved to {path_actor}, Critic to {path_critic}")

    def load(self, filename_actor=None, filename_critic=None):
        path_actor = filename_actor if filename_actor else global_config.PPO_ACTOR_MODEL_PATH
        path_critic = filename_critic if filename_critic else global_config.PPO_CRITIC_MODEL_PATH
        loaded_actor, loaded_critic = False, False
        if os.path.exists(path_actor):
            self.actor.load_state_dict(torch.load(path_actor, map_location=DEVICE))
            self.actor.eval(); loaded_actor = True
            print(f"PPO Actor loaded from {path_actor}")
        if os.path.exists(path_critic):
            self.critic.load_state_dict(torch.load(path_critic, map_location=DEVICE))
            self.critic.eval(); loaded_critic = True
            print(f"PPO Critic loaded from {path_critic}")
        if not (loaded_actor and loaded_critic):
             print(f"Error: PPO model (actor or critic or both) not found.")
             raise FileNotFoundError("PPO model files not found.")