# tetris_rl_agents/train.py
import argparse
import os
import random
import re # Added for regex in parsing filenames

import numpy as np
import torch
import config as tetris_config

from agents import AGENT_REGISTRY
from agents.ppo_agent import PPOAgent
# DQNAgent, REINFORCEAgent, A2CAgent, PPOAgent are implicitly used via AGENT_REGISTRY
from src.tetris import Tetris

# --- Helper Functions for Model Saving ---
def get_agent_file_prefix(agent_type_str, is_actor=False, is_critic=False):
    """Gets the filename prefix for the agent's model file."""
    processed_agent_type = agent_type_str.replace("_", "-") # e.g., dqn_original -> dqn-original
    if agent_type_str == "ppo": # PPO has distinct actor/critic prefixes
        if is_actor:
            return "ppo-actor"
        elif is_critic:
            return "ppo-critic"
        else: 
            return "ppo-model" # Fallback, should ideally not be hit
    return processed_agent_type

def parse_score_from_filename(filename_basename, expected_prefix):
    """
    Parses the score from a model filename.
    Expected format: {expected_prefix}_score_{score}.pth
    """
    # Regex to match the entire filename structure
    pattern = re.compile(f"^{re.escape(expected_prefix)}_score_(\\d+)\\.pth$")
    match = pattern.match(filename_basename)
    
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None

def find_best_existing_score(agent_prefix, model_dir):
    """
    Finds the highest score from existing model files for a given agent prefix in the model_dir.
    Returns the score (integer), or -1 if no relevant files are found, scores can't be parsed,
    or directory doesn't exist.
    """
    max_score = -1 
    if not os.path.isdir(model_dir):
        # Try to create if missing, as ensure_model_dir_exists might be called later
        try:
            os.makedirs(model_dir, exist_ok=True)
        except OSError:
            print(f"Warning: Model directory {model_dir} does not exist and could not be created.")
            return max_score 

    for filename in os.listdir(model_dir):
        score = parse_score_from_filename(filename, agent_prefix)
        if score is not None and score > max_score:
            max_score = score
    return max_score
# --- End Helper Functions ---

def get_args():
    parser = argparse.ArgumentParser(
        """Train Reinforcement Learning Agents to play Tetris""")
    
    parser.add_argument("--agent_type", type=str, default="dqn_original",
                        choices=list(tetris_config.AGENT_TYPES),
                        help="Type of agent to train.")
    parser.add_argument("--num_total_games", type=int, default=None,
                        help="Total number of games to play/learn from (overrides agent's config if set).")
    parser.add_argument("--render_game", action="store_true", help="Render the game during training.")
    
    args = parser.parse_args()
    return args

def train():
    opt = get_args()
    tetris_config.ensure_model_dir_exists() 

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
    
    total_games_to_play = opt.num_total_games
    if total_games_to_play is None:
        if opt.agent_type in ["genetic", "es"]:
            print(f"ERROR: Agent type '{opt.agent_type}' is evolutionary. Use 'train_evolutionary.py'.")
            return
        
        agent_config_key_lookup = opt.agent_type.upper()
        if agent_config_key_lookup == "DQN_ORIGINAL": 
            agent_config_key_lookup = "DQN" # Map to common DQN config for _TRAIN_GAMES
        
        config_train_games_var = f"{agent_config_key_lookup}_TRAIN_GAMES"
        
        if hasattr(tetris_config, config_train_games_var):
            total_games_to_play = getattr(tetris_config, config_train_games_var)
            print(f"--num_total_games not specified. Using config default: "
                  f"{total_games_to_play} games for '{opt.agent_type}' (from {config_train_games_var}).")
        else:
            print(f"Error: --num_total_games not specified, and configuration variable "
                  f"'{config_train_games_var}' not found in config.py for agent '{opt.agent_type}'.")
            print(f"Please specify --num_total_games or add '{config_train_games_var}' to config.py.")
            return

    EARLY_STOPPING_TARGET_SCORE = 1000000
    
    # Determine base directory for saving models (handles test_suite mode)
    current_model_base_dir = tetris_config.MODEL_DIR
    if 'test_suite' in tetris_config.__name__ and hasattr(tetris_config, 'PROJECT_ROOT'):
        current_model_base_dir = os.path.join(tetris_config.PROJECT_ROOT, "models_test_suite")
        if not os.path.exists(current_model_base_dir): # Ensure test suite model dir exists
            os.makedirs(current_model_base_dir, exist_ok=True)

    print(f"Starting training. Target games: {total_games_to_play}. "
          f"Early stopping if score >= {EARLY_STOPPING_TARGET_SCORE}.")
    print(f"Models will be saved to '{current_model_base_dir}' with filenames like "
          f"'{{agent-prefix}}_score_{{score}}.pth' upon achieving a new best score on disk.")


    s_t_board_features = env.reset()
    if tetris_config.DEVICE.type == 'cuda':
        s_t_board_features = s_t_board_features.cuda()
    
    pieces_played_count = 0
    games_played_count = 0
    current_game_score = 0 # Ensure this is int for filename compatibility
    total_score_all_games = 0.0
    last_s_prime_actual_features_for_ppo = None
    
    highest_score_this_session = -1 # Using -1 as baseline for scores (scores are non-negative)
    training_complete = False

    while games_played_count < total_games_to_play and not training_complete:
        action_tuple, aux_info = agent.select_action(s_t_board_features, env)
        
        reward, game_over = env.step(action_tuple, render=opt.render_game)
        
        pieces_played_count += 1
        current_game_score += int(reward) # Accumulate score as integer

        s_prime_actual_features = env.get_state_properties(env.board)
        if tetris_config.DEVICE.type == 'cuda':
            s_prime_actual_features = s_prime_actual_features.cuda()
        
        last_s_prime_actual_features_for_ppo = s_prime_actual_features

        agent.learn(
            state_features=s_t_board_features,
            action_tuple=action_tuple,
            reward=reward, 
            next_state_features=s_prime_actual_features,
            done=game_over,
            aux_info=aux_info
        )
        
        s_t_board_features = s_prime_actual_features

        if game_over:
            games_played_count += 1
            total_score_all_games += current_game_score
            avg_score = total_score_all_games / games_played_count if games_played_count > 0 else 0.0

            print_epsilon = ""
            if hasattr(agent, 'epsilon'):
                print_epsilon = f"| Epsilon: {agent.epsilon:.4f}"

            loss_str = "Loss: N/A"
            if hasattr(agent, 'last_loss') and agent.last_loss is not None:
                if isinstance(agent.last_loss, tuple) and len(agent.last_loss) == 2:
                    try: loss_str = f"Loss (A/C): {float(agent.last_loss[0]):.4f}/{float(agent.last_loss[1]):.4f}"
                    except: loss_str = "Loss (A/C): (err)"
                elif isinstance(agent.last_loss, dict):
                    parts = [f"{k.capitalize()}:{float(v_loss):.4f}" for k, v_loss in agent.last_loss.items() if isinstance(v_loss, (int, float))]
                    loss_str = f"Loss ({', '.join(parts)})" if parts else "Loss: (empty dict)"
                else:
                    try: loss_str = f"Loss: {float(agent.last_loss):.4f}"
                    except: loss_str = "Loss: (err)"
            elif hasattr(agent, 'actor_loss') and hasattr(agent, 'critic_loss') and \
                 agent.actor_loss is not None and agent.critic_loss is not None:
                 try: loss_str = f"Loss (A/C): {float(agent.actor_loss):.4f}/{float(agent.critic_loss):.4f}"
                 except: loss_str = "Loss (A/C): (err)"
            
            print(
                f"Game: {games_played_count}/{total_games_to_play} | "
                f"Score: {current_game_score} | Avg Score: {avg_score:.2f} | "
                f"Lines: {env.cleared_lines} {print_epsilon} | {loss_str}"
            )
            
            if current_game_score > highest_score_this_session:
                highest_score_this_session = current_game_score
                print(f"** New best score for this session: {highest_score_this_session}. Checking against disk. **")

                if opt.agent_type == "ppo":
                    actor_prefix = get_agent_file_prefix(opt.agent_type, is_actor=True)
                    critic_prefix = get_agent_file_prefix(opt.agent_type, is_critic=True)
                    disk_best_score = find_best_existing_score(actor_prefix, current_model_base_dir)

                    if current_game_score > disk_best_score:
                        print(f"Current score {current_game_score} > disk best PPO score {disk_best_score}. Saving.")
                        if disk_best_score > -1: # Remove old PPO files if they existed
                            old_actor_path = os.path.join(current_model_base_dir, f"{actor_prefix}_score_{disk_best_score}.pth")
                            old_critic_path = os.path.join(current_model_base_dir, f"{critic_prefix}_score_{disk_best_score}.pth")
                            if os.path.exists(old_actor_path): os.remove(old_actor_path)
                            if os.path.exists(old_critic_path): os.remove(old_critic_path)
                        
                        new_actor_path = os.path.join(current_model_base_dir, f"{actor_prefix}_score_{current_game_score}.pth")
                        new_critic_path = os.path.join(current_model_base_dir, f"{critic_prefix}_score_{current_game_score}.pth")
                        agent.save(new_actor_path, new_critic_path)
                        print(f"Saved PPO model (Score: {current_game_score}) to {new_actor_path} and {new_critic_path}")
                    else:
                        print(f"Session best {current_game_score}, but disk best PPO score is {disk_best_score}. Not overwriting disk.")
                
                elif opt.agent_type not in ["genetic", "es"]: # For non-PPO, non-evolutionary agents
                    agent_prefix = get_agent_file_prefix(opt.agent_type)
                    disk_best_score = find_best_existing_score(agent_prefix, current_model_base_dir)

                    if current_game_score > disk_best_score:
                        print(f"Current score {current_game_score} > disk best {opt.agent_type} score {disk_best_score}. Saving.")
                        if disk_best_score > -1: # Remove old model file if it existed
                            old_model_path = os.path.join(current_model_base_dir, f"{agent_prefix}_score_{disk_best_score}.pth")
                            if os.path.exists(old_model_path): os.remove(old_model_path)

                        new_model_path = os.path.join(current_model_base_dir, f"{agent_prefix}_score_{current_game_score}.pth")
                        agent.save(new_model_path)
                        print(f"Saved {opt.agent_type} model (Score: {current_game_score}) to {new_model_path}")
                    else:
                        print(f"Session best {current_game_score}, but disk best {opt.agent_type} score is {disk_best_score}. Not overwriting disk.")
            
            if current_game_score >= EARLY_STOPPING_TARGET_SCORE:
                print(f"\nEarly stopping: Target score of {EARLY_STOPPING_TARGET_SCORE} reached or exceeded "
                      f"with score {current_game_score} in game {games_played_count}.")
                training_complete = True

            if hasattr(agent, 'learn_episode') and callable(getattr(agent, 'learn_episode')):
                agent.learn_episode()
            agent.reset()
            
            s_t_board_features = env.reset()
            if tetris_config.DEVICE.type == 'cuda':
                s_t_board_features = s_t_board_features.cuda()
            current_game_score = 0 

        if isinstance(agent, PPOAgent) and hasattr(agent, 'memory_chosen_action_indices'):
            if not game_over and len(agent.memory_chosen_action_indices) >= agent.update_horizon:
                final_val_obs_for_ppo = last_s_prime_actual_features_for_ppo
                agent.learn_from_memory(final_val_obs_for_ppo)
            elif game_over and len(agent.memory_chosen_action_indices) > 0:
                agent.learn_from_memory(final_val_obs_for_ppo=None)

    print("\nTraining finished.")
    if training_complete and games_played_count > 0 and highest_score_this_session >= EARLY_STOPPING_TARGET_SCORE :
         pass 
    elif games_played_count >= total_games_to_play:
        print(f"Completed {games_played_count} games as per configuration.")
    
    if highest_score_this_session > -1:
        print(f"Highest score achieved in this training session: {highest_score_this_session}")
    elif games_played_count > 0 :
         print("No positive scores recorded in this session.")
    else:
        print("No games were completed in this session.")

if __name__ == "__main__":
    train()