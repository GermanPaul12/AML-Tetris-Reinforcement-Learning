# AML-TETRIS-RL/train_onpolicy.py

import os
import time
import argparse
import config as tetris_config

from agents import AGENT_REGISTRY
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

def train():
    opt = get_args()
    tetris_config.ensure_model_dir_exists()
    setup_seeds() # Set random seeds for reproducibility

    env = Tetris(
        width=tetris_config.GAME_WIDTH,
        height=tetris_config.GAME_HEIGHT,
        block_size=tetris_config.GAME_BLOCK_SIZE,
    )

    agent_class = AGENT_REGISTRY.get(opt.agent_type)
    if not agent_class:
        print(f"Error: Agent type '{opt.agent_type}' not found.")
        return

    print(f"\n--- Training Agent: {opt.agent_type.upper()} (On-Policy Style Loop) ---")
    agent = agent_class(state_size=tetris_config.STATE_SIZE, seed=tetris_config.SEED)

    # --- Determine training duration ---
    max_steps = opt.total_steps
    max_games = opt.num_games
    if max_steps is None and max_games is None:
        if opt.agent_type == "ppo":
            max_steps = getattr(tetris_config, "PPO_TOTAL_PIECES", 2000000)  # Increased default
            print(f"Using PPO_TOTAL_PIECES from config for max_steps: {max_steps}")
        elif opt.agent_type == "a2c":
            max_games = getattr(tetris_config, "A2C_TRAIN_GAMES", 10000)  # Increased default
            print(f"Using A2C_TRAIN_GAMES from config for max_games: {max_games}")
        else:
            max_steps = 2000000
            print(f"Warning: Using default max_steps: {max_steps}")

    EARLY_STOPPING_TARGET_SCORE = getattr(
        tetris_config, f"{opt.agent_type.upper()}_TARGET_GAME_SCORE", 2000000
    )  # Higher target

    current_model_base_dir = tetris_config.MODEL_DIR
    if "test_suite" in tetris_config.__name__ and hasattr(
        tetris_config, "PROJECT_ROOT"
    ):
        current_model_base_dir = os.path.join(
            tetris_config.PROJECT_ROOT, "models_test_suite"
        )
        if not os.path.exists(current_model_base_dir):
            os.makedirs(current_model_base_dir, exist_ok=True)

    print(
        f"Starting training. Max steps: {max_steps}, Max games: {max_games}. Early stopping score: {EARLY_STOPPING_TARGET_SCORE}."
    )
    print(f"Models will be saved to '{current_model_base_dir}' on new best score.")

    s_t_board_features = env.reset()
    if tetris_config.DEVICE.type == "cuda":
        s_t_board_features = s_t_board_features.cuda()

    current_total_steps = 0
    games_played_count = 0
    current_game_score = 0

    # For printing average scores
    total_score_for_avg = 0
    games_since_last_print = 0

    highest_score_this_session = -1  # Tracks best score in the current training run

    training_complete = False
    last_print_time = time.time()

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

        action_tuple, aux_info = agent.select_action(s_t_board_features, env)
        reward, game_over = env.step(action_tuple, render=opt.render_game)

        current_game_score += int(reward)
        current_total_steps += 1

        s_prime_actual_features = env.get_state_properties(env.board)
        if tetris_config.DEVICE.type == "cuda":
            s_prime_actual_features = s_prime_actual_features.cuda()

        agent.learn(
            state_features=s_t_board_features,
            action_tuple=action_tuple,
            reward=reward,
            next_state_features=s_prime_actual_features,
            done=game_over,
            aux_info=aux_info,
        )

        s_t_board_features = s_prime_actual_features

        if game_over:
            games_played_count += 1
            games_since_last_print += 1
            total_score_for_avg += current_game_score

            # PPO: Learn from any remaining buffer at episode end
            if opt.agent_type == "ppo" and hasattr(agent, "learn_on_episode_end"):
                agent.learn_on_episode_end()

            loss_str = "Loss: N/A"
            if hasattr(agent, "last_loss") and agent.last_loss is not None:
                if isinstance(agent.last_loss, tuple) and len(agent.last_loss) == 2:
                    try:
                        loss_str = f"Loss (A/C): {float(agent.last_loss[0]):.4f}/{float(agent.last_loss[1]):.4f}"
                    except Exception as e:
                        loss_str = f"Loss (A/C): ({e})"
                elif isinstance(agent.last_loss, dict):
                    parts = [
                        f"{k.capitalize()}:{float(v_loss):.4f}"
                        for k, v_loss in agent.last_loss.items()
                        if isinstance(v_loss, (int, float))
                    ]
                    loss_str = (
                        f"Loss ({', '.join(parts)})" if parts else "Loss: (empty dict)"
                    )
                else:
                    try:
                        loss_str = f"Loss: {float(agent.last_loss):.4f}"
                    except Exception as e:
                        loss_str = f"Loss: ({e})"

            # --- Print periodic average score ---
            if games_since_last_print >= opt.print_every_games:
                avg_score_since_last_print = (
                    total_score_for_avg / games_since_last_print
                    if games_since_last_print > 0
                    else 0
                )
                elapsed_time = time.time() - last_print_time
                steps_per_sec = (
                    (
                        current_total_steps
                        - (getattr(agent, "_last_print_steps", 0) or 0)
                    )
                    / elapsed_time
                    if elapsed_time > 0
                    else 0
                )
                agent._last_print_steps = current_total_steps  # Store for next SPS calc

                print(
                    f"Game: {games_played_count} | Steps: {current_total_steps} | "
                    f"Avg Score (last {games_since_last_print} games): {avg_score_since_last_print:.2f} | "
                    f"SPS: {steps_per_sec:.2f} | "
                    f"Score: {current_game_score} | Lines cleared: {env.cleared_lines} | {loss_str}"
                )
                total_score_for_avg = 0
                games_since_last_print = 0
                last_print_time = time.time()

            # --- Model Saving Logic ---
            if current_game_score > highest_score_this_session:
                highest_score_this_session = current_game_score
                print(
                    f"** New best score this session: {highest_score_this_session}. Checking against disk. **"
                )

                if opt.agent_type == "ppo":
                    actor_prefix = get_agent_file_prefix(opt.agent_type, is_actor=True)
                    critic_prefix = get_agent_file_prefix(
                        opt.agent_type, is_critic=True
                    )
                    # For PPO, we check against the score associated with the actor model on disk
                    disk_best_score = find_best_existing_score(
                        actor_prefix, current_model_base_dir
                    )

                    if current_game_score > disk_best_score:
                        print(
                            f"Current game score {current_game_score} > PPO disk best score {disk_best_score}. Saving."
                        )
                        if disk_best_score > -1:  # Remove old PPO files
                            old_actor_path = os.path.join(
                                current_model_base_dir,
                                f"{actor_prefix}_score_{disk_best_score}.pth",
                            )
                            old_critic_path = os.path.join(
                                current_model_base_dir,
                                f"{critic_prefix}_score_{disk_best_score}.pth",
                            )
                            if os.path.exists(old_actor_path):
                                os.remove(old_actor_path)
                            if os.path.exists(old_critic_path):
                                os.remove(old_critic_path)

                        new_actor_path = os.path.join(
                            current_model_base_dir,
                            f"{actor_prefix}_score_{current_game_score}.pth",
                        )
                        new_critic_path = os.path.join(
                            current_model_base_dir,
                            f"{critic_prefix}_score_{current_game_score}.pth",
                        )
                        agent.save(
                            new_actor_path, new_critic_path
                        )  # PPO agent's save method
                        print(
                            f"Saved PPO model (Score: {current_game_score}) to {new_actor_path} and {new_critic_path}"
                        )
                    else:
                        print(
                            f"Session best {current_game_score}, but PPO disk best is {disk_best_score}. Not overwriting."
                        )

                elif (
                    opt.agent_type == "a2c"
                ):  # A2C typically has one model or saves components similarly
                    agent_prefix = get_agent_file_prefix(opt.agent_type)
                    disk_best_score = find_best_existing_score(
                        agent_prefix, current_model_base_dir
                    )
                    if current_game_score > disk_best_score:
                        print(
                            f"Current game score {current_game_score} > A2C disk best score {disk_best_score}. Saving."
                        )
                        if disk_best_score > -1:
                            old_model_path = os.path.join(
                                current_model_base_dir,
                                f"{agent_prefix}_score_{disk_best_score}.pth",
                            )
                            if os.path.exists(old_model_path):
                                os.remove(old_model_path)
                        new_model_path = os.path.join(
                            current_model_base_dir,
                            f"{agent_prefix}_score_{current_game_score}.pth",
                        )
                        agent.save(new_model_path)  # A2C agent's save method
                        print(
                            f"Saved A2C model (Score: {current_game_score}) to {new_model_path}"
                        )
                    else:
                        print(
                            f"Session best {current_game_score}, but A2C disk best is {disk_best_score}. Not overwriting."
                        )
            # --- End Model Saving ---

            if current_game_score >= EARLY_STOPPING_TARGET_SCORE:
                print(
                    f"\nEarly stopping: Target score {EARLY_STOPPING_TARGET_SCORE} reached with {current_game_score}."
                )
                training_complete = True

            # Reset for next game
            agent.reset()
            current_game_score = 0
            s_t_board_features = env.reset()
            if tetris_config.DEVICE.type == "cuda":
                s_t_board_features = s_t_board_features.cuda()

    print("\nTraining finished.")
    if highest_score_this_session > -1:
        print(
            f"Highest score achieved in this training session: {highest_score_this_session}"
        )
    elif games_played_count > 0:
        print("No new high scores recorded in this session.")
    else:
        print("No games were completed in this session.")


if __name__ == "__main__":
    train()
