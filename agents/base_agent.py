# tetris_rl_agents/agents/base_agent.py
from abc import ABC, abstractmethod
import torch
from src.tetris import Tetris # For type hinting

class BaseAgent(ABC):
    def __init__(self, state_size):
        """
        Args:
            state_size (int): The number of features in the state representation.
        """
        self.state_size = state_size
        self.last_loss = None

    @abstractmethod
    def select_action(self, current_board_features: torch.Tensor, 
                      tetris_game_instance: Tetris, 
                      epsilon_override: float = None) -> tuple:
        """
        Selects an action based on the current board features and game instance.

        Args:
            current_board_features (torch.Tensor): Features of the board *before* current piece placement.
            tetris_game_instance (Tetris): The current game instance to query for next possible states.
            epsilon_override (float, optional): If provided, overrides the agent's internal epsilon (for exploration).

        Returns:
            tuple: (action_tuple, aux_info_dict)
                action_tuple (tuple): (x_pos, rotation_idx) representing the chosen piece placement.
                aux_info_dict (dict): Auxiliary information dictionary that might contain:
                    'features_s_prime_chosen': Features of the board state *after* the chosen action.
                    'current_board_features_s_t': Echo back of the input features.
                    'log_prob': Log probability of the chosen action (for policy gradient methods).
                    'entropy': Entropy of the policy (for policy gradient methods).
                    'value_of_current_board': Value of the current board state (for actor-critic methods).
                    'all_available_s_prime_features': List of features for all possible next states.
                    'chosen_action_index': Index of the chosen action among possible next states.
        """
        pass

    @abstractmethod
    def learn(self, state_features: torch.Tensor, action_tuple: tuple, reward: float, 
              next_state_features: torch.Tensor, done: bool, 
              game_instance_at_s = None, game_instance_at_s_prime = None, # For agents needing full game state
              aux_info: dict = None):
        """
        Primary learning step for agents that learn from individual transitions or need aux_info.
        Called by the training loop after each env.step().

        Args:
            state_features (torch.Tensor): Features of the board state S_t (before action_tuple).
            action_tuple (tuple): The action (placement) taken.
            reward (float): The reward R_{t+1} received after the action.
            next_state_features (torch.Tensor): Features of the board state S_{t+1} (after action and new piece appears).
            done (bool): True if the game ended after this transition.
            game_instance_at_s (Tetris, optional): Full Tetris game instance at state S_t.
            game_instance_at_s_prime (Tetris, optional): Full Tetris game instance at state S_{t+1}.
            aux_info (dict, optional): Auxiliary information dictionary returned by select_action.
        """
        pass
    
    @abstractmethod
    def reset(self):
        """
        Called at the start of each new game/episode by the training loop.
        Allows the agent to reset any internal episodic state.
        """
        pass # Default implementation does nothing.
    
    @abstractmethod
    def save(self, filename_primary=None, filename_secondary=None):
        """
        Saves the agent's model(s).
        PPO might use filename_primary for actor and filename_secondary for critic.
        Other agents might only use filename_primary.
        """
        pass

    @abstractmethod
    def load(self, filename_primary=None, filename_secondary=None):
        """
        Loads the agent's model(s).
        """
        pass