# tetris_rl_agents/agents/a2c_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import os

from src.tetris import Tetris  # For type hinting
import config as global_config
from .base_agent import BaseAgent

DEVICE = global_config.DEVICE


class ActorCriticNetwork(nn.Module):
    def __init__(
        self,
        state_size,
        seed=0,
        fc1_units=global_config.A2C_FC1_UNITS,
        fc2_units=global_config.A2C_FC2_UNITS,
    ):
        super(ActorCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Shared layers processing S_t or S' features
        self.fc_shared1 = nn.Linear(state_size, fc1_units)
        self.fc_shared2 = nn.Linear(fc1_units, fc2_units)

        # Actor head: Takes processed features (from S'), outputs a score for that S'
        self.actor_head = nn.Linear(fc2_units, 1)

        # Critic head: Takes processed features (from S_t), outputs V(S_t)
        self.critic_head = nn.Linear(fc2_units, 1)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def get_actor_scores(self, s_prime_features_batch):
        """Passes a batch of S' features through shared layers and actor head."""
        x = F.relu(self.fc_shared1(s_prime_features_batch))
        x_shared = F.relu(self.fc_shared2(x))
        return self.actor_head(x_shared)  # Scores for potential S' states

    def get_critic_value(self, s_t_features):
        """Passes S_t features through shared layers and critic head."""
        x = F.relu(self.fc_shared1(s_t_features))
        x_shared = F.relu(self.fc_shared2(x))
        return self.critic_head(x_shared)  # Value for current S_t


class A2CAgent(BaseAgent):
    def __init__(self, state_size, seed=0):
        super().__init__(state_size)
        self._agent_seed = seed
        random.seed(self._agent_seed)
        torch.manual_seed(self._agent_seed)
        if DEVICE.type == "cuda":
            torch.cuda.manual_seed_all(self._agent_seed)

        self.gamma = global_config.A2C_GAMMA
        self.entropy_coeff = global_config.A2C_ENTROPY_COEFF
        self.value_loss_coeff = global_config.A2C_VALUE_LOSS_COEFF
        learning_rate = global_config.A2C_LEARNING_RATE

        self.network = ActorCriticNetwork(state_size, seed=self._agent_seed).to(DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.last_loss = None  # Stores (actor_loss, critic_loss)

        print(
            f"A2C Agent initialized. LR: {learning_rate}, Gamma: {self.gamma}, Device: {DEVICE}"
        )
        print(
            f"  Entropy Coeff: {self.entropy_coeff}, Value Loss Coeff: {self.value_loss_coeff}"
        )

    def select_action(
        self,
        current_board_features_s_t: torch.Tensor,
        tetris_game_instance: Tetris,
        epsilon_override=None,
    ):  # Epsilon not used by A2C for exploration
        next_steps_dict = (
            tetris_game_instance.get_next_states()
        )  # {action_tuple: S'_features}
        if not next_steps_dict:
            # Fallback: should ideally not happen if game isn't over
            chosen_action_tuple = (tetris_game_instance.width // 2, 0)
            return chosen_action_tuple, {
                "log_prob": torch.tensor(0.0, device=DEVICE),
                "entropy": torch.tensor(0.0, device=DEVICE),
                "value_s_t": torch.tensor(0.0, device=DEVICE),  # Default/dummy value
                "features_s_prime_chosen": current_board_features_s_t.to(DEVICE),
                "current_board_features_s_t": current_board_features_s_t.to(DEVICE),
            }

        possible_actions_tuples = list(next_steps_dict.keys())
        s_prime_potential_features_list = [
            s_prime_feat.to(DEVICE) for s_prime_feat in next_steps_dict.values()
        ]
        s_prime_features_batch = torch.stack(s_prime_potential_features_list)

        self.network.train()

        action_scores = self.network.get_actor_scores(s_prime_features_batch).squeeze(
            -1
        )

        value_s_t = self.network.get_critic_value(
            current_board_features_s_t.unsqueeze(0).to(DEVICE)
        )

        dist = Categorical(logits=action_scores)
        chosen_idx_tensor = dist.sample()
        chosen_idx = chosen_idx_tensor.item()

        log_prob_chosen_action = dist.log_prob(chosen_idx_tensor)
        entropy = dist.entropy()

        chosen_action_tuple = possible_actions_tuples[chosen_idx]
        features_s_prime_chosen = s_prime_potential_features_list[chosen_idx]

        aux_info = {
            "log_prob": log_prob_chosen_action,
            "entropy": entropy,
            "value_s_t": value_s_t.squeeze(),
            "features_s_prime_chosen": features_s_prime_chosen,
            "current_board_features_s_t": current_board_features_s_t.to(DEVICE),
        }
        return chosen_action_tuple, aux_info

    # Corrected learn method signature to match train_onpolicy.py's call
    def learn(
        self,
        state_features: torch.Tensor,  # This is S_t (from s_t_board_features)
        action_tuple: tuple,
        reward: float,
        next_state_features: torch.Tensor,  # This is S_{t+1} (from s_prime_actual_features)
        done: bool,
        aux_info: dict = None,
    ):
        if not aux_info:
            self.last_loss = None
            return

        log_prob = aux_info["log_prob"]
        entropy = aux_info["entropy"]
        value_s_t_old = aux_info[
            "value_s_t"
        ]  # V(S_t) from selection time (aux_info['value_s_t'])

        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=DEVICE)

        value_s_t_plus_1 = torch.tensor([0.0], dtype=torch.float32, device=DEVICE)
        if not done:
            with torch.no_grad():
                self.network.eval()
                # Use 'next_state_features' which is S_{t+1}
                value_s_t_plus_1 = self.network.get_critic_value(
                    next_state_features.unsqueeze(0).to(DEVICE)
                ).squeeze()
                self.network.train()

        td_target = reward_tensor + self.gamma * value_s_t_plus_1 * (1.0 - float(done))
        advantage = (td_target - value_s_t_old).detach()

        actor_loss = -(log_prob * advantage).mean()

        # Re-evaluate V(S_t) for critic loss using current network parameters
        # Use 'state_features' which is S_t
        current_value_s_t_re_eval = self.network.get_critic_value(
            state_features.unsqueeze(0).to(DEVICE)
        ).squeeze()
        critic_loss = F.mse_loss(current_value_s_t_re_eval, td_target.detach())

        total_loss = (
            actor_loss
            + self.value_loss_coeff * critic_loss
            - self.entropy_coeff * entropy.mean()
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        self.last_loss = (actor_loss.item(), critic_loss.item())

    def reset(self):
        self.last_loss = None
        pass

    def save(self, filepath=None):
        path = filepath if filepath else global_config.A2C_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.network.state_dict(), path)
        print(f"A2C Agent saved to {path}")

    def load(self, filepath=None):
        path = filepath if filepath else global_config.A2C_MODEL_PATH
        if os.path.exists(path):
            self.network.load_state_dict(torch.load(path, map_location=DEVICE))
            self.network.train()
            print(f"A2C Agent loaded from {path}")
        else:
            print(f"Error: A2C model not found at {path}. Agent remains uninitialized.")
