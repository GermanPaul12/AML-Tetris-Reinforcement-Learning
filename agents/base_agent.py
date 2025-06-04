# tetris_rl_agents/agents/base_agent.py
from abc import ABC, abstractmethod
# No gymnasium needed if we pass state_size and action_size/info directly

class BaseAgent(ABC):
    def __init__(self, state_size: int): # Action space is dynamic in Tetris
        """
        Initializes the agent.

        Args:
            state_size (int): The dimensionality of the state space (4 for Tetris features).
            # action_space_info: Information about the action space, not a fixed gym.space
        """
        self.state_size = state_size
        # self.action_space_info = action_space_info # Store if needed by agent

    @abstractmethod
    def select_action(self, state_features, tetris_game_instance):
        """
        Selects an action tuple (x_pos, rotation_idx) based on the current state_features
        and the possible next moves from tetris_game_instance.

        Args:
            state_features (torch.Tensor): The current feature vector of the board state.
            tetris_game_instance (Tetris): The current instance of the Tetris game,
                                           used to call tetris_game_instance.get_next_states().
        Returns:
            action_tuple (tuple): The chosen (x_position, rotation_index).
            aux_info (dict, optional): Auxiliary information like log_probs for policy gradient methods.
        """
        pass

    def learn(self, state_features, action_tuple, reward, next_state_features, done,
              game_instance_at_s=None, game_instance_at_s_prime=None, aux_info=None):
        """
        Optional method for agents that learn from experience.

        Args:
            state_features: Features of the state S_t before the action.
            action_tuple: The action (A_t) taken (x_pos, rotation_idx).
            reward: The reward (R_{t+1}) received.
            next_state_features: Features of the state S_{t+1} after the action.
            done (bool): True if the game/episode ended.
            game_instance_at_s (Tetris, optional): Game state corresponding to state_features.
                                                    Needed if agent must re-evaluate actions from S_t.
            game_instance_at_s_prime (Tetris, optional): Game state corresponding to next_state_features.
                                                       Needed for target calculation in DQN/A2C/PPO.
            aux_info (dict, optional): Auxiliary info from select_action (e.g. log_probs).
        """
        pass

    def reset(self):
        """
        Optional: Resets agent's internal state at the start of a new game/episode.
        (e.g., for epsilon decay in DQN, or clearing buffers in REINFORCE).
        """
        pass

    def save(self, filepath):
        """Abstract method to save agent's model/parameters."""
        # Default can be pass, or raise NotImplementedError if all agents must save.
        # For PPO, filepath might be a prefix, and it saves actor and critic.
        # For GA/ES, it saves the best policy network.
        print(f"INFO: Base save called for {self.__class__.__name__}. Implement in child if needed.")
        pass


    def load(self, filepath):
        """Abstract method to load agent's model/parameters."""
        print(f"INFO: Base load called for {self.__class__.__name__}. Implement in child if needed.")
        pass