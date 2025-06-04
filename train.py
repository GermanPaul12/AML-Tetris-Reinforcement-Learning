# tetris_rl_agents/train.py
import argparse
import os
import random
import copy # For deep copying game state

import numpy as np
import torch

# Using the config name chosen by main.py or a default
# This logic should ideally be in main.py and config passed down,
# but for standalone train.py, we might need a way to switch.
# For now, assuming main.py sets up which config module is imported as 'tetris_config'
# If running train.py directly, it will use the default 'config.py'
try:
    if 'tetris_config_module_name' in os.environ:
        config_module_name = os.environ['tetris_config_module_name']
        if config_module_name.endswith(".py"):
            config_module_name = config_module_name[:-3]
        print(f"[train.py] Attempting to load config: {config_module_name}")
        tetris_config = __import__(config_module_name)
    else:
        import config as tetris_config
        print("[train.py] Loaded default config.py")
except ImportError:
    print("[train.py] Failed to load specified or default config. Exiting.")
    exit()


from agents import AGENT_REGISTRY
from agents.dqn_agent import DQNAgent
from agents.original_dqn_agent import OriginalDQNAgent
from agents.reinforce_agent import REINFORCEAgent
from agents.a2c_agent import A2CAgent
from agents.ppo_agent import PPOAgent
from src.tetris import Tetris

def get_args():
    parser = argparse.ArgumentParser(
        """Train Reinforcement Learning Agents to play Tetris""")
    
    parser.add_argument("--agent_type", type=str, default="dqn_original",
                        choices=list(tetris_config.AGENT_TYPES),
                        help="Type of agent to train.")
    parser.add_argument("--num_total_pieces", type=int, default=None,
                        help="Total number of pieces to play/learn from (overrides agent's config if set).")
    parser.add_argument("--render_game", action="store_true", help="Render the game during training.")
    parser.add_argument("--save_interval_pieces", type=int, default=5000, # Original was 1000 "epochs"
                        help="Save model every N pieces played.")
    
    args = parser.parse_args()
    return args

def train():
    opt = get_args()
    tetris_config.ensure_model_dir_exists() # Ensures MODEL_DIR from loaded config exists

    if torch.cuda.is_available():
        torch.cuda.manual_seed(tetris_config.SEED)
    else:
        torch.manual_seed(tetris_config.SEED)
    random.seed(tetris_config.SEED)
    np.random.seed(tetris_config.SEED)

    env = Tetris(width=tetris_config.GAME_WIDTH, height=tetris_config.GAME_HEIGHT,
                 block_size=tetris_config.GAME_BLOCK_SIZE)

    agent_class = AGENT_REGISTRY.get(opt.agent_type)
    if not agent_class:
        print(f"Error: Agent type '{opt.agent_type}' not found.")
        return
    print(f"\n--- Training Agent: {opt.agent_type.upper()} ---")
    agent = agent_class(state_size=tetris_config.STATE_SIZE, seed=tetris_config.SEED)
    
    total_pieces_to_play = opt.num_total_pieces
    if total_pieces_to_play is None:
        if opt.agent_type == "dqn" or opt.agent_type == "dqn_original":
            total_pieces_to_play = tetris_config.DQN_NUM_EPOCHS # This is the "num_epochs" from original
        elif opt.agent_type == "ppo":
            total_pieces_to_play = tetris_config.PPO_TOTAL_PIECES
        elif opt.agent_type in ["reinforce", "a2c"]:
            game_based_agent_train_games = getattr(tetris_config, f"{opt.agent_type.upper()}_TRAIN_GAMES", 500)
            avg_pieces_per_game_estimate = 200 # Rough estimate
            total_pieces_to_play = game_based_agent_train_games * avg_pieces_per_game_estimate
            print(f"Training {opt.agent_type} for approx. {game_based_agent_train_games} games (total piece limit: {total_pieces_to_play})")
        elif opt.agent_type in ["genetic", "es"]:
            print(f"ERROR: Agent type '{opt.agent_type}' is evolutionary. Use 'train_evolutionary.py'.")
            return
        else:
            total_pieces_to_play = tetris_config.MAX_EPOCHS_OR_PIECES
    
    print(f"Starting training for {total_pieces_to_play} piece placements...")

    # s_t_board_features: Features of the board *before* the current piece placement decision.
    # For the very first step, env.reset() gives this.
    s_t_board_features = env.reset() # Initial board state features
    if tetris_config.DEVICE.type == 'cuda':
        s_t_board_features = s_t_board_features.cuda()
    
    pieces_played_count = 0 # Equivalent to "epoch" in original script
    games_played_count = 0
    current_game_score = 0

    last_s_prime_actual_features_for_ppo = None # For PPO when horizon ends mid-game

    while pieces_played_count < total_pieces_to_play:
        # 1. Agent selects action.
        # `s_t_board_features` is passed. Agent uses `env` (tetris_game_instance) to call `get_next_states()`.
        # `aux_info` should contain `s_prime_chosen_features` for DQN-like agents.
        action_tuple, aux_info = agent.select_action(s_t_board_features, env)
        
        s_prime_chosen_features = aux_info.get('s_prime_chosen_features') # For DQN-like agents
        # For other agents like policy gradients, aux_info might contain log_probs etc.
        # Original DQN stored s_prime_chosen_features as "state" in replay.

        # 2. Environment takes a step based on `action_tuple`.
        reward, game_over = env.step(action_tuple, render=opt.render_game)
        
        pieces_played_count += 1
        current_game_score += reward

        # 3. Get S'_{actual} (s_prime_actual_features):
        # Features of the board *after* the chosen piece has landed AND a *new piece* has appeared.
        # This was "next_state" in the original DQN's replay buffer.
        s_prime_actual_features = env.get_state_properties(env.board)
        if tetris_config.DEVICE.type == 'cuda':
            s_prime_actual_features = s_prime_actual_features.cuda()
        
        last_s_prime_actual_features_for_ppo = s_prime_actual_features


        # 4. Agent learns.
        game_instance_at_s_prime = None # For DQNAgent (new one) to call get_next_states() from S'_{actual}
        if not game_over:
            # OriginalDQNAgent does NOT use game_instance_at_s_prime in its learn()
            # DQNAgent (new), A2C, PPO might use it for target value calculation.
            if isinstance(agent, DQNAgent) or \
               isinstance(agent, A2CAgent) or \
               isinstance(agent, PPOAgent):
                if not isinstance(agent, OriginalDQNAgent): # Exclude OriginalDQNAgent explicitly
                    game_instance_at_s_prime = copy.deepcopy(env)

        # For OriginalDQNAgent and DQNAgent, the `learn` method expects the "current state" for Q_expected
        # to be s_prime_chosen_features (passed via aux_info), and "next state" for Q_target
        # to be s_prime_actual_features.
        # For other agents, s_t_board_features is the "current state" S_t.
        
        current_state_for_learn = s_t_board_features # Default for PG, A2C, PPO critics
        if isinstance(agent, OriginalDQNAgent) or isinstance(agent, DQNAgent):
            # These agents' Q-learning update is based on Q(S'_chosen) vs R + gamma * max_a Q(S'_{actual}, a)
            # Their learn methods are structured to use aux_info['s_prime_chosen_features'] internally
            # and next_state_features (which is s_prime_actual_features).
            # So, the first arg state_features (s_t_board_features) is context, not the Q-learning state.
            pass # Their internal learn logic will use aux_info correctly.

        agent.learn(
            state_features=current_state_for_learn, # S_t for PG/A2C/PPO critic, context for DQN-likes
            action_tuple=action_tuple,
            reward=reward,
            next_state_features=s_prime_actual_features, # S'_{actual}
            done=game_over,
            game_instance_at_s=None, 
            game_instance_at_s_prime=game_instance_at_s_prime,
            aux_info=aux_info
        )
        
        # 5. The state for the *next iteration's* call to `agent.select_action`
        #    (i.e., the next `s_t_board_features`) is `s_prime_actual_features`.
        s_t_board_features = s_prime_actual_features

        # 6. Handle game over
        if game_over:
            games_played_count += 1
            print_epsilon = ""
            if hasattr(agent, 'epsilon'): # Check if agent has epsilon (DQNs)
                # For OriginalDQNAgent, num_epochs_elapsed is internal & drives its epsilon
                # For DQNAgent, current_epoch_for_decay is internal
                print_epsilon = f"| Epsilon: {agent.epsilon:.4f}"
            
            print(
                f"Pieces: {pieces_played_count}/{total_pieces_to_play} | "
                f"Game: {games_played_count} | Score: {current_game_score} | "
                f"Tetrominoes: {env.tetrominoes} | Lines: {env.cleared_lines} {print_epsilon}"
            )
            
            if isinstance(agent, REINFORCEAgent):
                agent.learn_episode()

            agent.reset() # Agent's internal game reset (e.g. REINFORCE buffers)
            
            s_t_board_features = env.reset() # Env reset for new game
            if tetris_config.DEVICE.type == 'cuda':
                s_t_board_features = s_t_board_features.cuda()
            current_game_score = 0

        # 7. PPO specific learning
        if isinstance(agent, PPOAgent):
            if len(agent.memory_chosen_action_indices) >= agent.update_horizon:
                final_val_obs_for_ppo = None if game_over else last_s_prime_actual_features_for_ppo
                agent.learn_from_memory(final_val_obs_for_ppo)

        # 8. Save model periodically
        if pieces_played_count > 0 and pieces_played_count % opt.save_interval_pieces == 0:
            model_save_path = getattr(tetris_config, f"{opt.agent_type.upper()}_MODEL_PATH", None)
            if opt.agent_type == "dqn_original":
                model_save_path = tetris_config.ORIGINAL_DQN_MODEL_PATH
            
            if opt.agent_type == "ppo":
                agent.save(tetris_config.PPO_ACTOR_MODEL_PATH, tetris_config.PPO_CRITIC_MODEL_PATH)
            elif model_save_path:
                agent.save(model_save_path)
            else:
                default_save_dir = tetris_config.MODEL_DIR
                if 'test_suite' in tetris_config.__name__: # If using test config
                    default_save_dir = os.path.join(tetris_config.PROJECT_ROOT, "models_test_suite")
                agent.save(os.path.join(default_save_dir, f"{opt.agent_type}_tetris_pieces_{pieces_played_count}.pth"))

    print("\nTraining finished.")
    final_model_save_path = getattr(tetris_config, f"{opt.agent_type.upper()}_MODEL_PATH", None)
    if opt.agent_type == "dqn_original":
         final_model_save_path = tetris_config.ORIGINAL_DQN_MODEL_PATH
    
    if opt.agent_type == "ppo":
        agent.save(tetris_config.PPO_ACTOR_MODEL_PATH, tetris_config.PPO_CRITIC_MODEL_PATH)
    elif final_model_save_path:
        agent.save(final_model_save_path)
    else:
        default_save_dir = tetris_config.MODEL_DIR
        if 'test_suite' in tetris_config.__name__:
            default_save_dir = os.path.join(tetris_config.PROJECT_ROOT, "models_test_suite")
        agent.save(os.path.join(default_save_dir, f"{opt.agent_type}_tetris_final.pth"))

if __name__ == "__main__":
    # This helps select config if train.py is run directly for debugging
    # Set environment variable before running:
    # export TETRIS_CONFIG_MODULE=config_test_suite
    # Or uncomment one of these lines in code:
    # os.environ['tetris_config_module_name'] = 'config' # For normal config
    # os.environ['tetris_config_module_name'] = 'config_test_suite' # For test config
    
    # The import at the top will handle loading based on this env var if set.
    # If not set, it defaults to 'config'.
    train()