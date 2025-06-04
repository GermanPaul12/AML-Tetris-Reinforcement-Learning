# tetris_rl_agents/agents/random_agent.py
import random
import torch # For type hinting state_features
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from .base_agent import BaseAgent
# from src.tetris import Tetris # For type hinting tetris_game_instance

class RandomAgent(BaseAgent):
    def __init__(self, state_size, seed=0):
        super().__init__(state_size)
        random.seed(seed)
        print("RandomAgent for Tetris initialized.")

    def select_action(self, state_features: torch.Tensor, tetris_game_instance):
        """
        Selects a random action from the possible next moves.
        """
        next_steps_dict = tetris_game_instance.get_next_states()
        possible_actions_tuples = list(next_steps_dict.keys())

        if not possible_actions_tuples:
            # Fallback, should ideally not happen if game is ongoing
            # Default action: attempt to place piece in middle with no rotation
            # print("Warning: RandomAgent found no possible actions. Returning default.")
            chosen_action_tuple = (tetris_game_instance.width // 2, 0)
            # Create a dummy feature vector for aux_info if necessary, or an empty dict
            # For random agent, aux_info might not be strictly needed by trainer,
            # but if trainer expects it, provide a valid structure.
            # Here, 'features_after_chosen_action' is for DQN primarily.
            # For simplicity, random agent can return None or empty dict for aux_info.
            return chosen_action_tuple, {} # No aux_info needed
        
        chosen_action_tuple = random.choice(possible_actions_tuples)
        
        # For consistency, aux_info can be an empty dict if not used
        return chosen_action_tuple, {} 