# tetris_rl_agents/agents/ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
import os

from .base_agent import BaseAgent
import config as global_config

DEVICE = global_config.DEVICE


# --- Networks (Your existing ActorPPO and CriticPPO are perfect for this) ---
class ActorPPO(nn.Module):
    def __init__(
        self,
        state_size,
        seed=0,
        fc1_units=global_config.PPO_ACTOR_FC1,
        fc2_units=global_config.PPO_ACTOR_FC2,
    ):
        super(ActorPPO, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc_actor_head = nn.Linear(fc2_units, 1)  # Scores a single S'
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, s_prime_features_batch):
        x = F.relu(self.fc1(s_prime_features_batch))
        x = F.relu(self.fc2(x))
        return self.fc_actor_head(x)


class CriticPPO(nn.Module):
    def __init__(
        self,
        state_size,
        seed=0,
        fc1_units=global_config.PPO_CRITIC_FC1,
        fc2_units=global_config.PPO_CRITIC_FC2,
    ):
        super(CriticPPO, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc_critic = nn.Linear(fc2_units, 1)  # Outputs V(S_t)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, s_t_features):
        x = F.relu(self.fc1(s_t_features))
        x = F.relu(self.fc2(x))
        return self.fc_critic(x)


# --- PPO Agent ---
class PPOAgent(BaseAgent):
    def __init__(self, state_size, seed=0):
        super().__init__(state_size)
        # Hyperparameters from config
        self.gamma = global_config.PPO_GAMMA
        self.gae_lambda = global_config.PPO_GAE_LAMBDA
        self.ppo_epochs = global_config.PPO_EPOCHS_PER_UPDATE
        self.ppo_clip = global_config.PPO_CLIP_EPSILON
        self.entropy_coeff = global_config.PPO_ENTROPY_COEFF
        self.value_loss_coeff = global_config.PPO_VALUE_LOSS_COEFF
        self.batch_size = global_config.PPO_BATCH_SIZE
        self.update_horizon = global_config.PPO_UPDATE_HORIZON

        # Actor and Critic networks
        self.actor = ActorPPO(state_size, seed=seed).to(DEVICE)
        self.critic = CriticPPO(state_size, seed=seed).to(DEVICE)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=global_config.PPO_ACTOR_LR
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=global_config.PPO_CRITIC_LR
        )

        # Buffer to store trajectory data
        self.memory = {
            "s_t_features": [],  # S_t
            "all_s_prime_features_lists": [],  # List of all possible S' from S_t
            "chosen_action_indices": [],  # Index of chosen action in the list above
            "log_probs_old": [],  # Log prob of the chosen action
            "rewards": [],  # Reward received
            "dones": [],  # Done flag
            "values_s_t": [],  # V(S_t) from critic
        }
        self.last_loss = {}  # For logging multiple losses

        print(
            f"PPO Agent initialized. Update Horizon: {self.update_horizon}, Device: {DEVICE}"
        )

    def select_action(
        self, current_board_features_s_t, tetris_game_instance, epsilon_override=None
    ):
        next_steps_dict = tetris_game_instance.get_next_states()
        if not next_steps_dict:  # Game over or stuck
            return (
                tetris_game_instance.width // 2,
                0,
            ), {}  # Return default action, empty aux_info

        all_s_prime_features_list = [
            s_prime_feat.to(DEVICE) for s_prime_feat in next_steps_dict.values()
        ]
        all_s_prime_features_batch = torch.stack(all_s_prime_features_list)

        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            action_scores = self.actor(all_s_prime_features_batch).squeeze(-1)
            value_s_t = self.critic(current_board_features_s_t.unsqueeze(0).to(DEVICE))
        self.actor.train()
        self.critic.train()

        action_probs = F.softmax(action_scores, dim=0)
        dist = Categorical(action_probs)
        chosen_action_index_tensor = dist.sample()
        chosen_action_index = chosen_action_index_tensor.item()

        log_prob_tensor = dist.log_prob(chosen_action_index_tensor)
        entropy_tensor = dist.entropy()

        chosen_action_tuple = list(next_steps_dict.keys())[chosen_action_index]

        aux_info = {
            "log_prob": log_prob_tensor,
            "entropy": entropy_tensor,
            "value_s_t": value_s_t.squeeze(),
            "all_available_s_prime_features": all_s_prime_features_list,
            "chosen_action_index": chosen_action_index,
        }
        return chosen_action_tuple, aux_info

    def learn(
        self, state_features, action_tuple, reward, next_state_features, done, aux_info
    ):
        # Called after every step by train_onpolicy.py
        # Store the transition in memory
        if not aux_info:
            return  # Can't learn without aux_info

        self.memory["s_t_features"].append(state_features.cpu())
        self.memory["all_s_prime_features_lists"].append(
            [f.cpu() for f in aux_info["all_available_s_prime_features"]]
        )
        self.memory["chosen_action_indices"].append(aux_info["chosen_action_index"])
        self.memory["log_probs_old"].append(aux_info["log_prob"].cpu())
        self.memory["values_s_t"].append(aux_info["value_s_t"].cpu())
        self.memory["rewards"].append(reward)
        self.memory["dones"].append(done)

        # If buffer is full, trigger the update
        if len(self.memory["rewards"]) >= self.update_horizon:
            self.learn_from_memory(
                next_state_features
            )  # Pass the final next_state for bootstrapping

    def learn_on_episode_end(self):
        # Called by train_onpolicy.py when an episode ends, to process any remaining data
        if len(self.memory["rewards"]) > 0:
            self.learn_from_memory(
                last_s_t_plus_1_features=None
            )  # No bootstrapping at terminal state

    def learn_from_memory(self, last_s_t_plus_1_features=None):
        # --- 1. Calculate Advantages and Returns (Targets for Critic) ---
        rewards = torch.tensor(self.memory["rewards"], dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(self.memory["dones"], dtype=torch.float32).to(DEVICE)
        values = torch.tensor(self.memory["values_s_t"], dtype=torch.float32).to(DEVICE)

        advantages = []
        gae = 0.0

        # Determine value of the state after the last action in the trajectory
        with torch.no_grad():
            if last_s_t_plus_1_features is not None:  # Horizon ended mid-game
                last_value = self.critic(
                    last_s_t_plus_1_features.unsqueeze(0).to(DEVICE)
                ).squeeze()
            else:  # Horizon ended because game was over
                last_value = torch.tensor(0.0, device=DEVICE)

        # GAE calculation (iterating backwards)
        for i in reversed(range(len(rewards))):
            next_value = values[i + 1] if i + 1 < len(values) else last_value
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        advantages = torch.stack(advantages)
        returns_for_critic = (
            advantages + values
        )  # TD(lambda) returns as targets for V(S_t)

        # Normalize advantages (common practice)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 2. Convert other memory lists to tensors ---
        old_log_probs = torch.tensor(
            self.memory["log_probs_old"], dtype=torch.float32
        ).to(DEVICE)
        old_s_t_features = torch.stack(self.memory["s_t_features"]).to(DEVICE)
        old_all_s_prime_lists_cpu = self.memory["all_s_prime_features_lists"]
        old_chosen_action_indices = torch.tensor(
            self.memory["chosen_action_indices"], dtype=torch.long
        ).to(DEVICE)

        num_samples = len(rewards)
        sample_indices = np.arange(num_samples)

        # --- 3. Perform PPO Update for K Epochs ---
        for _ in range(self.ppo_epochs):
            np.random.shuffle(sample_indices)
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = sample_indices[start:end]

                # Get minibatch data
                mb_s_t = old_s_t_features[batch_idx]
                mb_all_s_prime_lists_cpu = [
                    old_all_s_prime_lists_cpu[i] for i in batch_idx
                ]
                mb_chosen_indices = old_chosen_action_indices[batch_idx]
                mb_old_log_probs = old_log_probs[batch_idx]
                mb_advantages = advantages[batch_idx]
                mb_returns = returns_for_critic[batch_idx]

                # Evaluate old actions with current policy
                new_log_probs_list, entropy_list = [], []
                for i in range(len(batch_idx)):
                    s_prime_features_batch = torch.stack(
                        mb_all_s_prime_lists_cpu[i]
                    ).to(DEVICE)

                    action_scores = self.actor(s_prime_features_batch).squeeze(-1)
                    dist = Categorical(
                        logits=action_scores
                    )  # Use logits for numerical stability

                    new_log_probs_list.append(dist.log_prob(mb_chosen_indices[i]))
                    entropy_list.append(dist.entropy())

                new_log_probs = torch.stack(new_log_probs_list)
                entropy = torch.stack(entropy_list).mean()

                # --- Actor Loss (Clipped Objective) ---
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip)
                    * mb_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- Critic Loss (Value Loss) ---
                new_values = self.critic(mb_s_t).squeeze()
                critic_loss = F.mse_loss(new_values, mb_returns)

                # --- Total Loss and Update ---
                total_loss = (
                    actor_loss
                    + self.value_loss_coeff * critic_loss
                    - self.entropy_coeff * entropy
                )

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        # Log losses for monitoring
        self.last_loss = {
            "actor": actor_loss.item(),
            "critic": critic_loss.item(),
            "entropy": entropy.item(),
        }

        # Clear memory for the next trajectory collection
        self.clear_memory()

    def clear_memory(self):
        for key in self.memory:
            self.memory[key] = []

    def save(self, filename_actor=None, filename_critic=None):
        path_actor = (
            filename_actor if filename_actor else global_config.PPO_ACTOR_MODEL_PATH
        )
        path_critic = (
            filename_critic if filename_critic else global_config.PPO_CRITIC_MODEL_PATH
        )
        os.makedirs(os.path.dirname(path_actor), exist_ok=True)
        os.makedirs(os.path.dirname(path_critic), exist_ok=True)
        torch.save(self.actor.state_dict(), path_actor)
        torch.save(self.critic.state_dict(), path_critic)
        print(f"PPO Actor saved to {path_actor}, Critic to {path_critic}")

    def load(self, filename_actor=None, filename_critic=None):
        path_actor = (
            filename_actor if filename_actor else global_config.PPO_ACTOR_MODEL_PATH
        )
        path_critic = (
            filename_critic if filename_critic else global_config.PPO_CRITIC_MODEL_PATH
        )
        if os.path.exists(path_actor):
            self.actor.load_state_dict(torch.load(path_actor, map_location=DEVICE))
        if os.path.exists(path_critic):
            self.critic.load_state_dict(torch.load(path_critic, map_location=DEVICE))
        self.actor.train()
        self.critic.train()  # Set to train mode for further training
        print(f"PPO models loaded. Actor: {path_actor}, Critic: {path_critic}")
