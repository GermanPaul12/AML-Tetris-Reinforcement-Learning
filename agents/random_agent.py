# tetris_rl_agents/agents/random_agent.py
import random
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, state_size, seed=0):
        super().__init__(state_size)
        random.seed(seed)
        print("RandomAgent for Tetris initialized.")

    def select_action(
        self, state_features: torch.Tensor, tetris_game_instance, epsilon_override=0.0
    ):
        """
        Selects a random action from the possible next moves.
        """
        next_steps_dict = tetris_game_instance.get_next_states()
        possible_actions_tuples = list(next_steps_dict.keys())

        if not possible_actions_tuples:
            chosen_action_tuple = (tetris_game_instance.width // 2, 0)
            return chosen_action_tuple, {}

        chosen_action_tuple = random.choice(possible_actions_tuples)

        # For consistency, aux_info can be an empty dict if not used
        return chosen_action_tuple, {}
