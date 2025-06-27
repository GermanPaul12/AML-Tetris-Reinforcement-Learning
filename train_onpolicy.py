import os
import argparse
import config as tetris_config

from agents import PPOAgent, A2CAgent

from helper import *
from src.tetris import Tetris

def get_args() -> argparse.Namespace:
    """Parses command-line arguments for training configuration."""
    parser = argparse.ArgumentParser("""Train A2C or PPO Agents for Tetris""")
    parser.add_argument("--agent_type", type=str, default="ppo", choices=["a2c", "ppo"])
    parser.add_argument("--total_steps", type=int, default=None, help="Total environment steps to train for.")
    parser.add_argument("--num_games", type=int, default=None, help="Alternative: Total number of games to complete training.")
    parser.add_argument("--render_game", action="store_true", help="Render the game.")
    parser.add_argument("--print_every_games", type=int, default=10, help="Frequency of printing average scores (in games).")
    return parser.parse_args()

#################################################
# Main Training Function for A2C and PPO Agents #
#################################################

def train(opt: argparse.Namespace):
    """Train an A2C or PPO agent for Tetris using an on-policy style loop."""

    agent_type = opt.agent_type.lower()
    max_steps = opt.total_steps
    max_games = opt.num_games
    target_score = tetris_config.SCORE_TARGET
    current_model_base_dir = tetris_config.MODEL_DIR
    
    if agent_type == "ppo":
        print("\n--- Training PPO Agent ---")
        ppo_controller = PPOAgent(state_size=tetris_config.STATE_SIZE, seed=tetris_config.SEED)
        
        max_steps = tetris_config.PPO_TOTAL_PIECES or max_steps
        print(f"Using PPO_TOTAL_PIECES from config for max_steps: {max_steps}")
    elif agent_type == "a2c":
        print("\n--- Training A2C Agent ---")
        a2c_controller = A2CAgent(state_size=tetris_config.STATE_SIZE, seed=tetris_config.SEED)
        
        max_games = tetris_config.A2C_TRAIN_GAMES or max_games
        print(f"Using A2C_TRAIN_GAMES from config for max_games: {max_games}")
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}. Choose 'a2c' or 'ppo'.")

    print(f"Starting training. Max steps: {max_steps}, Max games: {max_games}. Early stopping score: {target_score}.")
    print(f"Models will be saved to '{current_model_base_dir}' on new best score.")

    # === Start Training Loop ===
    env = Tetris()
    s_t_board_features = env.reset()
    if tetris_config.DEVICE.type == "cuda": s_t_board_features = s_t_board_features.cuda()

    current_total_steps = 0
    games_played_count = 0
    current_game_score = 0

    # For printing average scores
    total_score_for_avg = 0
    total_lines_cleared_for_avg = 0
    games_since_last_print = 0
    highest_score_this_session = -1
    
    training_complete = False

    while not training_complete:
        # --- Check termination conditions ---
        if max_steps and current_total_steps >= max_steps:
            print(f"\nReached max_steps: {current_total_steps}")
            training_complete = True
            break
        if max_games and games_played_count >= max_games:
            print(f"\nReached max_games: {games_played_count}")
            training_complete = True
            break
        
        if agent_type == "ppo": action_tuple, aux_info = ppo_controller.select_action(s_t_board_features, env)
        else: action_tuple, aux_info = a2c_controller.select_action(s_t_board_features, env)
        
        reward, game_over = env.step(action_tuple, render=opt.render_game)
        current_game_score += reward
        current_total_steps += 1

        s_prime_actual_features = env.get_state_properties(env.board)
        if tetris_config.DEVICE.type == "cuda": s_prime_actual_features = s_prime_actual_features.cuda()

        if agent_type == "ppo":
            ppo_controller.learn(
                state_features=s_t_board_features,
                reward=reward,
                next_state_features=s_prime_actual_features,
                done=game_over,
                aux_info=aux_info
            )
        else:
            a2c_controller.learn(
                reward=reward,
                next_state_features=s_prime_actual_features,
                done=game_over,
                aux_info=aux_info
            )

        s_t_board_features = s_prime_actual_features
        
        if game_over:
            games_played_count += 1
            games_since_last_print += 1
            total_score_for_avg += current_game_score
            total_lines_cleared_for_avg += env.cleared_lines

            if agent_type == "ppo":
                ppo_controller.learn_on_episode_end()
                parts = [f"{k.capitalize()}:{float(v_loss):.4f}" for k, v_loss in ppo_controller.last_loss.items()]
                loss_str = (f"Loss ({', '.join(parts)})" if parts else "Loss: (empty dict)")
            else:
                loss_str = f"Loss (A/C): {float(a2c_controller.last_loss[0]):.4f}/{float(a2c_controller.last_loss[1]):.4f}"

            # --- Print periodic average score ---
            if games_since_last_print >= opt.print_every_games:
                total_lines_cleared_for_avg = total_lines_cleared_for_avg // games_since_last_print if games_since_last_print > 0 else 0
                avg_score_since_last_print = (total_score_for_avg / games_since_last_print if games_since_last_print > 0 else 0)

                print(
                    f"Game: {games_played_count} | Steps: {current_total_steps} | "
                    f"Avg Score (last {games_since_last_print} games): {avg_score_since_last_print:.2f} | "
                    f"Avg Lines cleared (last {games_since_last_print} games): {total_lines_cleared_for_avg} | {loss_str}"
                )
                total_score_for_avg = 0
                total_lines_cleared_for_avg = 0
                games_since_last_print = 0

            # --- Model Saving Logic ---
            if current_game_score > highest_score_this_session:
                highest_score_this_session = current_game_score
                print(f"** New best score this session: {highest_score_this_session}. Checking against disk. **")

                agent_prefix = get_agent_file_prefix(agent_type)
                disk_best_score = find_best_existing_score(agent_prefix, current_model_base_dir)

                if current_game_score > disk_best_score:
                    print(f"Current game score {current_game_score} > {agent_prefix.upper()} disk best score {disk_best_score}. Saving.")
                    if disk_best_score > -1:
                        old_model_path = os.path.join(current_model_base_dir, f"{agent_prefix}_score_{disk_best_score}.pth")
                        if os.path.exists(old_model_path): os.remove(old_model_path)
                    new_model_path = os.path.join(current_model_base_dir, f"{agent_prefix}_score_{current_game_score}.pth")
                    if agent_type == "ppo": ppo_controller.save(new_model_path)
                    else: a2c_controller.save(new_model_path)
                    print(f"Saved {agent_prefix.upper()} model (Score: {current_game_score}) to {new_model_path}")
                else:
                    print(f"Session best {current_game_score}, but {agent_prefix.upper()} disk best is {disk_best_score}. Not overwriting.")
            # --- End Model Saving ---

            if current_game_score >= target_score:
                print(f"\nEarly stopping: Target score {target_score} reached with {current_game_score}.")
                training_complete = True

            # Reset for next game
            if agent_type == "ppo": ppo_controller.reset()
            else: a2c_controller.reset()
            
            current_game_score = 0
            s_t_board_features = env.reset()
            if tetris_config.DEVICE.type == "cuda": s_t_board_features = s_t_board_features.cuda()

    print("\nTraining finished.")
    if highest_score_this_session > -1: print(f"Highest score achieved in this training session: {highest_score_this_session}")
    elif games_played_count > 0: print("No new high scores recorded in this session.")
    else: print("No games were completed in this session.")


if __name__ == "__main__":
    opt = get_args()
    tetris_config.ensure_model_dir_exists()
    setup_seeds()
    
    train(opt)
