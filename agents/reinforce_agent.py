# tetris_rl_agents/agents/reinforce_agent.py
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import config
from .base_agent import BaseAgent, PolicyNetwork

DEVICE = config.DEVICE

class REINFORCEAgent(BaseAgent):
    def __init__(self, state_size, seed:int=0):
        super().__init__(state_size)
        self.policy_network = PolicyNetwork(state_size, fc1_units=config.REINFORCE_FC1_UNITS, fc2_units=config.REINFORCE_FC2_UNITS, seed=seed).to(DEVICE)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=config.REINFORCE_LEARNING_RATE)
        self.gamma = config.REINFORCE_GAMMA

        self.saved_log_probs = []
        self.rewards = []
        self.episodes_done = 0
        self.last_loss = None

        print("REINFORCE Agent initialized to score S' states.")

    def select_action(self, current_board_features_s_t, tetris_game_instance, epsilon_override=None):
        next_steps_dict = tetris_game_instance.get_next_states()

        if not next_steps_dict:
            chosen_action_tuple = (tetris_game_instance.width // 2, 0)
            features_s_prime_chosen = current_board_features_s_t
            return chosen_action_tuple, {
                "features_s_prime_chosen": features_s_prime_chosen.to(DEVICE),
                "current_board_features_s_t": current_board_features_s_t.to(DEVICE),
                "log_prob": torch.tensor(0.0).to(DEVICE),  # Placeholder, but indicates issue
                "entropy": torch.tensor(0.0).to(DEVICE),  # Placeholder
            }

        possible_actions_tuples = list(next_steps_dict.keys())
        s_prime_potential_features_list = [feat.to(DEVICE) for feat in next_steps_dict.values()]
        s_prime_features_batch = torch.stack(s_prime_potential_features_list)

        self.policy_network.train()
        action_scores = self.policy_network(s_prime_features_batch).squeeze(-1) # Remove last dim, grad_fn
        if action_scores.dim() == 0:  action_scores = action_scores.unsqueeze(0)  # Make it [1]

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

    def expand_memory(self, reward, state_info):
        if state_info and "log_prob" in state_info:
            self.rewards.append(reward)
            self.saved_log_probs.append(state_info["log_prob"])
        else:
            print("Warning: REINFORCEAgent.learn() called without log_prob in state_info.")

    def learn_episode(self):
        if not self.saved_log_probs:  # Nothing to learn if no actions were taken/logged
            self.last_loss = None
            return

        returns = []
        discounted_reward = 0
        for r in reversed(self.rewards):
            discounted_reward = r + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []
        for log_prob, R_t in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R_t)  # No .item() here, keep as tensor

        self.optimizer.zero_grad()
        total_loss = torch.stack(policy_loss).sum()
        total_loss.backward()
        self.optimizer.step()

        self.last_loss = total_loss.item()
        self.reset()

    def reset(self):
        self.rewards = []
        self.saved_log_probs = []

    def save(self, filename_primary=None, filename_secondary=None):
        path = (filename_primary if filename_primary else config.REINFORCE_MODEL_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "policy_network_state_dict": self.policy_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "episodes_done": self.episodes_done,  # Save episodes if tracked
            }, path)
        print(f"REINFORCE Agent saved to {path}")

    def load(self, filename_primary=None, filename_secondary=None):
        path = (filename_primary if filename_primary else config.REINFORCE_MODEL_PATH)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            self.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.episodes_done = checkpoint.get("episodes_done", 0)
            self.policy_network.train()  # Set to train mode
            print(f"REINFORCE Agent loaded from {path}. Episodes trained: {self.episodes_done}")
        else:
            print(f"ERROR: No REINFORCE model found at {path}.")
