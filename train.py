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
    
    # Keep --agent_type, but default might change if focusing on DQN
    parser.add_argument("--agent_type", type=str, default="dqn", # Assuming 'dqn' is your V-learning agent
                        choices=list(tetris_config.AGENT_TYPES),
                        help="Type of agent to train.")
    
    # This will now refer to total "learning epochs" (game-over learning steps)
    parser.add_argument("--num_epochs", type=int, default=None, 
                        help="Total number of learning epochs (game-over learning steps). Overrides agent's config.")
    parser.add_argument("--render_game", action="store_true", help="Render the game during training.")
    
    args = parser.parse_args()
    return args

def train():
    opt = get_args()
    tetris_config.ensure_model_dir_exists() 

    # --- Seeding ---
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
        
    if opt.agent_type not in ["dqn"]: # This loop is tailored for DQN original style
        print(f"Warning: This train.py script's epsilon and learning step logic is optimized for DQN "
              f"V-learning style (learn on game over). Agent '{opt.agent_type}' may behave differently.")

    print(f"\n--- Training Agent: {opt.agent_type.upper()} (Original Style Loop with Epsilon Control) ---")
    agent = agent_class(state_size=tetris_config.STATE_SIZE, seed=tetris_config.SEED)
    
    # --- Determine Total Learning Epochs (from command line or config) ---
    total_learning_epochs = opt.num_epochs
    if total_learning_epochs is None:
        # For DQN, use DQN_TOTAL_LEARNING_UPDATES from config (should be e.g., 3000)
        total_learning_epochs = getattr(tetris_config, "DQN_TOTAL_LEARNING_UPDATES", 3000)
        print(f"--num_epochs not specified. Using DQN_TOTAL_LEARNING_UPDATES from config: {total_learning_epochs} learning epochs.")
    
    # --- Epsilon Parameters (from config, to be used by this script) ---
    initial_epsilon = tetris_config.DQN_EPSILON_START
    final_epsilon = tetris_config.DQN_EPSILON_MIN
    # num_decay_epochs here is the number of *learning steps* for decay
    num_decay_learning_steps = tetris_config.DQN_EPSILON_DECAY_EPOCHS 

    EARLY_STOPPING_TARGET_SCORE = 1000000 
    
    current_model_base_dir = tetris_config.MODEL_DIR
    if 'test_suite' in tetris_config.__name__ and hasattr(tetris_config, 'PROJECT_ROOT'):
        current_model_base_dir = os.path.join(tetris_config.PROJECT_ROOT, "models_test_suite")
        if not os.path.exists(current_model_base_dir):
            os.makedirs(current_model_base_dir, exist_ok=True)

    print(f"Starting training. Target learning epochs: {total_learning_epochs}. ")
    print(f"Epsilon will decay from {initial_epsilon} to {final_epsilon} over {num_decay_learning_steps} learning epochs.")
    print(f"Models will be saved to '{current_model_base_dir}'.")

    s_t_board_features = env.reset()
    if tetris_config.DEVICE.type == 'cuda':
        s_t_board_features = s_t_board_features.cuda()
    
    current_epoch = 0 # This is the learning step counter, like original `epoch`
    games_played_count = 0
    current_game_score = 0
    total_score_all_games = 0.0
    
    highest_score_this_session = -1
    training_complete = False

    while current_epoch < total_learning_epochs and not training_complete:
        
        # Calculate epsilon for this step, based on `current_epoch` (learning steps done)
        # This mirrors the original train.py precisely
        current_epsilon_for_action = final_epsilon + \
            (max(num_decay_learning_steps - current_epoch, 0) * 
             (initial_epsilon - final_epsilon) / num_decay_learning_steps) \
            if num_decay_learning_steps > 0 else final_epsilon
        
        # Pass calculated epsilon to agent for action selection
        action_tuple, aux_info = agent.select_action(
            s_t_board_features, 
            env, 
            epsilon_override=current_epsilon_for_action
        )
        
        reward, game_over = env.step(action_tuple, render=opt.render_game)
        current_game_score += int(reward)

        s_prime_actual_features = env.get_state_properties(env.board)
        if tetris_config.DEVICE.type == 'cuda':
            s_prime_actual_features = s_prime_actual_features.cuda()
        
        # Agent's learn method now primarily adds to memory
        agent.learn( 
            state_features=s_t_board_features,
            action_tuple=action_tuple,        
            reward=reward,                    
            next_state_features=s_prime_actual_features, 
            done=game_over,                   
            aux_info=aux_info                 
        )
        
        if game_over:
            games_played_count += 1
            total_score_all_games += current_game_score
            avg_score = total_score_all_games / games_played_count if games_played_count > 0 else 0.0

            # --- Learning Step ---
            can_learn = False
            if hasattr(agent, 'memory') and agent.memory is not None:
                # Original condition: len(replay_memory) < opt.replay_memory_size / 10
                if len(agent.memory) >= (tetris_config.DQN_BUFFER_SIZE / 10) and \
                   len(agent.memory) >= tetris_config.DQN_BATCH_SIZE :
                    can_learn = True
            
            if can_learn:
                experiences = agent.memory.sample()
                if experiences[0].size(0) >= agent.memory.batch_size:
                    agent._learn_from_experiences(experiences, tetris_config.DQN_GAMMA)
                    # Agent's internal counter for learning steps done, useful for saving/loading agent state
                    if hasattr(agent, 'learning_steps_done'):
                        agent.learning_steps_done += 1 
                
                # This `current_epoch` is the one driving epsilon and total training duration
                current_epoch += 1 
            # --- End Learning Step ---

            # Epsilon is now controlled by train.py, so agent.epsilon might not be primary
            # For printing, use the value calculated by train.py
            print_epsilon = f"| Epsilon: {current_epsilon_for_action:.4f}"
            
            loss_str = "Loss: N/A"
            if hasattr(agent, 'last_loss') and agent.last_loss is not None:
                # ... (loss formatting as before) ...
                 try: loss_str = f"Loss: {float(agent.last_loss):.4f}"
                 except: loss_str = "Loss: (err)"

            print(
                f"Epoch: {current_epoch}/{total_learning_epochs} | Game: {games_played_count} | "
                f"Score: {current_game_score} | Avg Score: {avg_score:.2f} | "
                f"Lines: {env.cleared_lines} {print_epsilon} | {loss_str}"
            )
            
            # --- Model Saving Logic ---
            if current_game_score > highest_score_this_session:
                highest_score_this_session = current_game_score
                print(f"** New best score for this session: {highest_score_this_session}. Checking against disk. **")
                # Simplified saving for DQN (V-learning agent)
                if opt.agent_type == "dqn":
                    agent_prefix = get_agent_file_prefix(opt.agent_type)
                    disk_best_score = find_best_existing_score(agent_prefix, current_model_base_dir)
                    if current_game_score > disk_best_score:
                        print(f"Current score {current_game_score} > disk best {opt.agent_type} score {disk_best_score}. Saving.")
                        if disk_best_score > -1: 
                            old_model_path = os.path.join(current_model_base_dir, f"{agent_prefix}_score_{disk_best_score}.pth")
                            if os.path.exists(old_model_path): os.remove(old_model_path)
                        new_model_path = os.path.join(current_model_base_dir, f"{agent_prefix}_score_{current_game_score}.pth")
                        agent.save(new_model_path) # Agent's save method
                        print(f"Saved {opt.agent_type} model (Score: {current_game_score}) to {new_model_path}")
                    else:
                        print(f"Session best {current_game_score}, but disk best {opt.agent_type} score is {disk_best_score}. Not overwriting disk.")
                # Add elif for PPO or other agents if they have different saving needs
            # --- End Model Saving ---
            
            if current_game_score >= EARLY_STOPPING_TARGET_SCORE:
                print(f"\nEarly stopping: Target score {EARLY_STOPPING_TARGET_SCORE} reached.")
                training_complete = True

            if hasattr(agent, 'learn_episode') and callable(getattr(agent, 'learn_episode')):
                agent.learn_episode() 
            agent.reset() 
            
            current_game_score = 0 
            s_t_board_features = env.reset() 
            if tetris_config.DEVICE.type == 'cuda':
                s_t_board_features = s_t_board_features.cuda()
        else: 
            s_t_board_features = s_prime_actual_features
        
    print("\nTraining finished.")
    # ... (final print statements as before) ...
    if training_complete and current_epoch > 0 : # Check current_epoch as it indicates learning happened
         pass 
    elif current_epoch >= total_learning_epochs:
        print(f"Completed {current_epoch} learning epochs as per configuration.")
    
    if highest_score_this_session > -1:
        print(f"Highest score achieved in this training session: {highest_score_this_session}")
    elif games_played_count > 0 : # games_played_count is still relevant for avg score
         print("No new high scores recorded in this session (or learning did not occur).")
    else:
        print("No games or learning epochs were completed in this session.")
        
        
if __name__ == "__main__":
    train()