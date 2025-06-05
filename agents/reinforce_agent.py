# tetris_rl_agents/agents/reinforce_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
import os

from .base_agent import BaseAgent
import config as global_config  # Your project's config

DEVICE = global_config.DEVICE


# Policy Network for REINFORCE - Actor scores S' states
class PolicyNetworkREINFORCE(nn.Module):
    def __init__(self, state_size, fc1_units, fc2_units):
        super(PolicyNetworkREINFORCE, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3_scores = nn.Linear(fc2_units, 1)  # Outputs a raw score for an S' state

    def forward(self, s_prime_features_batch):  # Takes features of potential S' states
        x = F.relu(self.fc1(s_prime_features_batch))
        x = F.relu(self.fc2(x))
        scores = self.fc3_scores(x)  # Raw scores for each S'
        return scores


class REINFORCEAgent(BaseAgent):
    def __init__(self, state_size, seed=0):
        super().__init__(state_size)
        self.policy_network = PolicyNetworkREINFORCE(
            state_size,
            global_config.REINFORCE_FC1_UNITS,
            global_config.REINFORCE_FC2_UNITS,
        ).to(DEVICE)
        self.optimizer = optim.Adam(
            self.policy_network.parameters(), lr=global_config.REINFORCE_LEARNING_RATE
        )
        self.gamma = global_config.REINFORCE_GAMMA

        self.saved_log_probs = []
        self.rewards = []
        self.episodes_done = 0  # Optional counter
        self.last_loss = None

        # Seed for reproducibility
        torch.manual_seed(seed)
        if DEVICE.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        print("REINFORCE Agent initialized to score S' states.")

    def select_action(
        self, current_board_features_s_t, tetris_game_instance, epsilon_override=None
    ):
        # epsilon_override is ignored by REINFORCE
        next_steps_dict = tetris_game_instance.get_next_states()

        if not next_steps_dict:
            # Fallback if no actions (should mean game over)
            chosen_action_tuple = (tetris_game_instance.width // 2, 0)
            features_s_prime_chosen = current_board_features_s_t
            # For REINFORCE, if no actions, we can't form a distribution.
            # This scenario should ideally be handled by the training loop not calling select_action.
            # If forced, log_prob would be problematic.
            return chosen_action_tuple, {
                "features_s_prime_chosen": features_s_prime_chosen.to(DEVICE),
                "current_board_features_s_t": current_board_features_s_t.to(DEVICE),
                "log_prob": torch.tensor(0.0).to(
                    DEVICE
                ),  # Placeholder, but indicates issue
                "entropy": torch.tensor(0.0).to(DEVICE),  # Placeholder
            }

        possible_actions_tuples = list(next_steps_dict.keys())
        s_prime_potential_features_list = [
            feat.to(DEVICE) for feat in next_steps_dict.values()
        ]

        s_prime_features_batch = torch.stack(s_prime_potential_features_list)

        self.policy_network.train()  # Ensure network is in train mode for dropout/batchnorm if any, though not typical for this simple MLP
        action_scores = self.policy_network(s_prime_features_batch).squeeze(
            -1
        )  # Remove last dim if it's [N, 1]

        # Handle case of single action to avoid squeeze error with softmax if action_scores is scalar
        if action_scores.dim() == 0:  # If only one action, scores is a scalar
            action_scores = action_scores.unsqueeze(0)  # Make it [1]

        action_probs = F.softmax(action_scores, dim=0)
        dist = Categorical(action_probs)
        chosen_idx_tensor = dist.sample()
        chosen_idx = chosen_idx_tensor.item()

        log_prob_chosen_action = dist.log_prob(chosen_idx_tensor)
        entropy = dist.entropy()

        chosen_action_tuple = possible_actions_tuples[chosen_idx]
        features_s_prime_chosen = s_prime_potential_features_list[chosen_idx]

        return chosen_action_tuple, {
            "features_s_prime_chosen": features_s_prime_chosen,
            "current_board_features_s_t": current_board_features_s_t.to(DEVICE),
            "log_prob": log_prob_chosen_action,
            "entropy": entropy,  # Though REINFORCE doesn't typically use entropy bonus
        }

    def learn(
        self, state_features, action_tuple, reward, next_state_features, done, aux_info
    ):
        # This method is called after every step by train_dqn_reinforce.py
        # It stores the reward and the log_prob of the action taken.
        if aux_info and "log_prob" in aux_info:
            self.rewards.append(reward)
            self.saved_log_probs.append(aux_info["log_prob"])
        else:
            print(
                "Warning: REINFORCEAgent.learn() called without log_prob in aux_info."
            )

    def learn_episode(self):
        # Called at the end of an episode by train_dqn_reinforce.py
        if not self.saved_log_probs:  # Nothing to learn if no actions were taken/logged
            self.last_loss = None
            return

        returns = []
        discounted_reward = 0
        for r in reversed(self.rewards):
            discounted_reward = r + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
        # Normalize returns (optional but often helps)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []
        for log_prob, R_t in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R_t)  # No .item() here, keep as tensor

        self.optimizer.zero_grad()
        # Summing up losses for all steps in the episode
        total_loss = torch.stack(policy_loss).sum()
        total_loss.backward()
        self.optimizer.step()

        self.last_loss = total_loss.item()
        self.reset()  # Clear buffers for the next episode

    def reset(self):
        self.rewards = []
        self.saved_log_probs = []
        # self.last_loss = None # Reset in train_dqn_reinforce.py on game_over after printing

    def save(self, filename_primary=None, filename_secondary=None):
        path = (
            filename_primary if filename_primary else global_config.REINFORCE_MODEL_PATH
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "policy_network_state_dict": self.policy_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "episodes_done": self.episodes_done,  # Save episodes if tracked
            },
            path,
        )
        print(f"REINFORCE Agent saved to {path}")

    def load(self, filename_primary=None, filename_secondary=None):
        path = (
            filename_primary if filename_primary else global_config.REINFORCE_MODEL_PATH
        )
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            self.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.episodes_done = checkpoint.get("episodes_done", 0)
            self.policy_network.train()  # Set to train mode
            print(
                f"REINFORCE Agent loaded from {path}. Episodes trained: {self.episodes_done}"
            )
        else:
            print(f"ERROR: No REINFORCE model found at {path}.")
