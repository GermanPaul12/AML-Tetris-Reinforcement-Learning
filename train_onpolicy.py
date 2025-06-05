# tetris_rl_agents/train_onpolicy.py
import argparse
import os
import random
import re

import numpy as np
import torch
import config as tetris_config  # Your project's config

from agents import AGENT_REGISTRY
from agents.ppo_agent import PPOAgent  # For PPO specific logic if needed
from src.tetris import Tetris


def get_agent_file_prefix(agent_type_str, is_actor=False, is_critic=False):
    processed_agent_type = agent_type_str.replace("_", "-")
    if agent_type_str == "ppo":
        if is_actor:
            return "ppo-actor"
        elif is_critic:
            return "ppo-critic"
        else:
            return "ppo-model"
    return processed_agent_type


def parse_score_from_filename(filename_basename, expected_prefix):
    pattern = re.compile(f"^{re.escape(expected_prefix)}_score_(\\d+)\\.pth$")
    match = pattern.match(filename_basename)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def find_best_existing_score(agent_prefix, model_dir):
    max_score = -1
    if not os.path.isdir(model_dir):
        try:
            os.makedirs(model_dir, exist_ok=True)
        except OSError:
            print(
                f"Warning: Model directory {model_dir} does not exist and could not be created."
            )
            return max_score
    for filename in os.listdir(model_dir):
        score = parse_score_from_filename(filename, agent_prefix)
        if score is not None and score > max_score:
            max_score = score
    return max_score


def get_args():
    parser = argparse.ArgumentParser("""Train A2C or PPO Agents for Tetris""")

    parser.add_argument(
        "--agent_type",
        type=str,
        default="a2c",
        choices=["a2c", "ppo"],  # Restrict choices
        help="Type of on-policy agent to train (a2c or ppo).",
    )

    # Different ways to specify training duration for these agents
    parser.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help="Total environment steps to train for.",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=None,
        help="Total number of games to complete training.",
    )
    parser.add_argument("--render_game", action="store_true", help="Render the game.")

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

    # Determine training duration
    # Prioritize total_steps, then num_games, then agent's config (e.g., PPO_TOTAL_PIECES)
    max_steps = opt.total_steps
    max_games = opt.num_games

    if max_steps is None and max_games is None:
        if opt.agent_type == "ppo":
            max_steps = getattr(tetris_config, "PPO_TOTAL_PIECES", 1000000)
            print(f"Using PPO_TOTAL_PIECES from config for max_steps: {max_steps}")
        elif opt.agent_type == "a2c":
            max_games = getattr(
                tetris_config, "A2C_TRAIN_GAMES", 5000
            )  # A2C might use games
            print(f"Using A2C_TRAIN_GAMES from config for max_games: {max_games}")
        else:  # Fallback
            max_steps = 1000000
            print(f"Using default max_steps: {max_steps}")

    EARLY_STOPPING_TARGET_SCORE = getattr(
        tetris_config, f"{opt.agent_type.upper()}_TARGET_GAME_SCORE", 1000000
    )

    current_model_base_dir = tetris_config.MODEL_DIR
    # ... (test_suite dir handling)

    print(
        f"Starting training. Max steps: {max_steps}, Max games: {max_games}. Early stopping: {EARLY_STOPPING_TARGET_SCORE}."
    )
    print(f"Models will be saved to '{current_model_base_dir}'.")

    s_t_board_features = env.reset()
    if tetris_config.DEVICE.type == "cuda":
        s_t_board_features = s_t_board_features.cuda()

    current_total_steps = 0
    games_played_count = 0
    current_game_score = 0

    highest_score_this_session = -1
    training_complete = False

    while not training_complete:
        if max_steps is not None and current_total_steps >= max_steps:
            print(f"Reached max_steps: {max_steps}")
            training_complete = True
            break
        if max_games is not None and games_played_count >= max_games:
            print(f"Reached max_games: {max_games}")
            training_complete = True
            break

        # For on-policy agents, epsilon_override is typically not used.
        action_tuple, aux_info = agent.select_action(s_t_board_features, env)

        reward, game_over = env.step(action_tuple, render=opt.render_game)
        current_game_score += int(reward)
        current_total_steps += 1

        s_prime_actual_features = env.get_state_properties(env.board)
        if tetris_config.DEVICE.type == "cuda":
            s_prime_actual_features = s_prime_actual_features.cuda()

        # Agent learns per step (A2C) or adds to buffer (PPO)
        # PPO's learn method might trigger an update if its horizon is met.
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

            # PPO specific: might learn from remaining buffer at episode end
            if isinstance(agent, PPOAgent):  # Check if PPOAgent has a method for this
                if hasattr(agent, "learn_on_episode_end") and callable(
                    getattr(agent, "learn_on_episode_end")
                ):
                    agent.learn_on_episode_end()
                elif (
                    hasattr(agent, "memory")
                    and len(agent.memory) > 0
                    and hasattr(agent, "learn_from_memory")
                    and callable(getattr(agent, "learn_from_memory"))
                ):
                    # A more generic call if PPO stores data and processes it
                    print(
                        f"PPO: Processing remaining buffer at end of game {games_played_count}"
                    )
                    agent.learn_from_memory(
                        final_val_obs=None
                    )  # Assuming this is the interface

            loss_str = "Loss: N/A"
            # A2C/PPO might have actor_loss, critic_loss in agent.last_loss or separate attributes
            if hasattr(agent, "last_loss") and agent.last_loss is not None:
                if (
                    isinstance(agent.last_loss, tuple) and len(agent.last_loss) == 2
                ):  # (actor_loss, critic_loss)
                    try:
                        loss_str = f"Loss (A/C): {float(agent.last_loss[0]):.4f}/{float(agent.last_loss[1]):.4f}"
                    except:
                        loss_str = "Loss (A/C): (err)"
                elif isinstance(
                    agent.last_loss, dict
                ):  # For PPO storing multiple losses
                    parts = [
                        f"{k.capitalize()}:{float(v_loss):.4f}"
                        for k, v_loss in agent.last_loss.items()
                        if isinstance(v_loss, (int, float))
                    ]
                    loss_str = (
                        f"Loss ({', '.join(parts)})" if parts else "Loss: (empty dict)"
                    )
                else:  # Single loss value
                    try:
                        loss_str = f"Loss: {float(agent.last_loss):.4f}"
                    except:
                        loss_str = "Loss: (err)"

            # PPO might print stats based on its update cycles, not just game end
            # For simplicity, we print per game here.
            print(
                f"Game: {games_played_count} | Steps: {current_total_steps} | "
                f"Score: {current_game_score} | Lines: {env.cleared_lines} | {loss_str}"
            )

            # --- Model Saving Logic (adapted for PPO) ---
            if current_game_score > highest_score_this_session:
                highest_score_this_session = current_game_score
                print(
                    f"** New best score for this session: {highest_score_this_session}. Checking against disk. **"
                )
                if opt.agent_type == "ppo":
                    actor_prefix = get_agent_file_prefix(opt.agent_type, is_actor=True)
                    critic_prefix = get_agent_file_prefix(
                        opt.agent_type, is_critic=True
                    )
                    disk_best_score_actor = find_best_existing_score(
                        actor_prefix, current_model_base_dir
                    )
                    # PPO saving might be tied to a single score for both actor/critic
                    # For simplicity, using actor_prefix score as the reference.
                    if current_game_score > disk_best_score_actor:
                        print(
                            f"Current PPO score {current_game_score} > disk best {disk_best_score_actor}. Saving."
                        )
                        if disk_best_score_actor > -1:
                            old_actor_path = os.path.join(
                                current_model_base_dir,
                                f"{actor_prefix}_score_{disk_best_score_actor}.pth",
                            )
                            old_critic_path = os.path.join(
                                current_model_base_dir,
                                f"{critic_prefix}_score_{disk_best_score_actor}.pth",
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
                        agent.save(new_actor_path, new_critic_path)  # PPO agent's save
                        print(
                            f"Saved PPO model (Score: {current_game_score}) to {new_actor_path} & {new_critic_path}"
                        )
                elif (
                    opt.agent_type == "a2c"
                ):  # A2C usually has one model or saves components similarly
                    agent_prefix = get_agent_file_prefix(opt.agent_type)
                    disk_best_score = find_best_existing_score(
                        agent_prefix, current_model_base_dir
                    )
                    if current_game_score > disk_best_score:
                        print(
                            f"Current A2C score {current_game_score} > disk best {disk_best_score}. Saving."
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
                        agent.save(new_model_path)  # A2C agent's save
                        print(
                            f"Saved A2C model (Score: {current_game_score}) to {new_model_path}"
                        )
            # --- End Model Saving ---

            if current_game_score >= EARLY_STOPPING_TARGET_SCORE:
                print(
                    f"\nEarly stopping: Target score {EARLY_STOPPING_TARGET_SCORE} reached."
                )
                training_complete = True

            agent.reset()
            current_game_score = 0
            s_t_board_features = env.reset()
            if tetris_config.DEVICE.type == "cuda":
                s_t_board_features = s_t_board_features.cuda()

    print("\nTraining finished.")


if __name__ == "__main__":
    train()
