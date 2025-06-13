import torch
import random

from .base_agent import BaseAgent
from src.tetris import Tetris

class RandomAgent(BaseAgent):
    def __init__(self, state_size, seed=0):
        super().__init__(state_size)
        random.seed(seed)
        print("RandomAgent for Tetris initialized.")

    def select_action(self, state_features:torch.Tensor, tetris_game_instance:Tetris, epsilon_override=0.0):
        """ Selects a random action from the possible next moves.
        Args:
            state_features (torch.Tensor): The current state features of the Tetris game.
            tetris_game_instance (Tetris): The Tetris game instance to interact with.
            epsilon_override (float): Not used in this agent, but kept for compatibility.
        Returns:
            tuple: A tuple representing the chosen action (x, rotation).
            dict: An empty dictionary, as no additional information is needed for this agent.
        """
        next_steps_dict = tetris_game_instance.get_next_states()
        possible_actions_tuples = list(next_steps_dict.keys())

        if not possible_actions_tuples:
            chosen_action_tuple = (tetris_game_instance.width // 2, 0)
            return chosen_action_tuple, {}

        chosen_action_tuple = random.choice(possible_actions_tuples)

        return chosen_action_tuple, {}
