# tetris_rl_agents/train_dqn_reinforce.py
import argparse
import os
import random
import re

import numpy as np
import torch
import config as tetris_config

from agents import AGENT_REGISTRY
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
    parser = argparse.ArgumentParser("""Train DQN or REINFORCE Agents for Tetris""")

    parser.add_argument(
        "--agent_type",
        type=str,
        default="dqn",
        choices=["dqn", "reinforce"],
        help="Type of agent to train (dqn or reinforce).",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Total number of learning epochs (game-over learning steps for DQN, or completed episodes for REINFORCE).",
    )
    parser.add_argument(
        "--render_game", action="store_true", help="Render the game during training."
    )

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

    print(
        f"\n--- Training Agent: {opt.agent_type.upper()} (DQN/REINFORCE Style Loop) ---"
    )
    agent = agent_class(state_size=tetris_config.STATE_SIZE, seed=tetris_config.SEED)

    total_learning_epochs = opt.num_epochs
    if total_learning_epochs is None:
        if opt.agent_type == "dqn":
            total_learning_epochs = getattr(
                tetris_config, "DQN_TOTAL_LEARNING_UPDATES", 3000
            )
        elif opt.agent_type == "reinforce":
            total_learning_epochs = getattr(
                tetris_config, "REINFORCE_TRAIN_GAMES", 5000
            )  # REINFORCE uses TRAIN_GAMES as epochs
        else:
            total_learning_epochs = 3000  # Fallback
        print(
            f"--num_epochs not specified. Using config default: {total_learning_epochs} learning epochs/episodes."
        )

    # Epsilon parameters (primarily for DQN)
    initial_epsilon = tetris_config.DQN_EPSILON_START
    final_epsilon = tetris_config.DQN_EPSILON_MIN
    num_decay_learning_steps = tetris_config.DQN_EPSILON_DECAY_EPOCHS

    EARLY_STOPPING_TARGET_SCORE = 1000000

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
        f"Starting training. Target learning epochs/episodes: {total_learning_epochs}."
    )
    if opt.agent_type == "dqn":
        print(
            f"DQN Epsilon will decay from {initial_epsilon} to {final_epsilon} over {num_decay_learning_steps} learning epochs."
        )
    print(f"Models will be saved to '{current_model_base_dir}'.")

    s_t_board_features = env.reset()
    if tetris_config.DEVICE.type == "cuda":
        s_t_board_features = s_t_board_features.cuda()

    current_epoch = 0
    games_played_this_run = 0
    current_game_score = 0
    total_score_all_games = 0.0

    highest_score_this_session = -1
    training_complete = False

    while current_epoch < total_learning_epochs and not training_complete:
        epsilon_for_action_selection = None
        if opt.agent_type == "dqn":
            epsilon_for_action_selection = (
                final_epsilon
                + (
                    max(num_decay_learning_steps - current_epoch, 0)
                    * (initial_epsilon - final_epsilon)
                    / num_decay_learning_steps
                )
                if num_decay_learning_steps > 0
                else final_epsilon
            )

        action_tuple, aux_info = agent.select_action(
            s_t_board_features,
            env,
            epsilon_override=epsilon_for_action_selection,  # REINFORCE will ignore this
        )

        reward, game_over = env.step(action_tuple, render=opt.render_game)
        current_game_score += int(reward)

        s_prime_actual_features = env.get_state_properties(env.board)
        if tetris_config.DEVICE.type == "cuda":
            s_prime_actual_features = s_prime_actual_features.cuda()

        # Agent's learn method stores experience.
        # For REINFORCE, it stores (reward, log_prob). For DQN, (S_t, S'_chosen, R, D).
        agent.learn(
            state_features=s_t_board_features,
            action_tuple=action_tuple,
            reward=reward,
            next_state_features=s_prime_actual_features,
            done=game_over,
            aux_info=aux_info,
        )

        if game_over:
            games_played_this_run += 1
            total_score_all_games += current_game_score
            avg_score = (
                total_score_all_games / games_played_this_run
                if games_played_this_run > 0
                else 0.0
            )

            # --- Learning Step specific to agent type ---
            learned_this_epoch = False
            if opt.agent_type == "dqn":
                can_learn_dqn = False
                if hasattr(agent, "memory") and agent.memory is not None:
                    if len(agent.memory) >= tetris_config.DQN_BATCH_SIZE and len(
                        agent.memory
                    ) >= (tetris_config.DQN_BUFFER_SIZE / 10):
                        can_learn_dqn = True

                if can_learn_dqn:
                    experiences = agent.memory.sample()
                    if experiences[0].size(0) >= agent.memory.batch_size:
                        agent._learn_from_experiences(
                            experiences, tetris_config.DQN_GAMMA
                        )
                        if hasattr(
                            agent, "learning_steps_done"
                        ):  # DQN internal counter
                            agent.learning_steps_done += 1
                        learned_this_epoch = True

            elif opt.agent_type == "reinforce":
                if hasattr(agent, "learn_episode") and callable(
                    getattr(agent, "learn_episode")
                ):
                    agent.learn_episode()  # REINFORCE learns from the whole episode
                    if hasattr(
                        agent, "episodes_done"
                    ):  # REINFORCE might track episodes
                        agent.episodes_done += 1
                    learned_this_epoch = True

            if learned_this_epoch:
                current_epoch += 1
            # --- End Learning Step ---

            print_epsilon_str = ""
            if opt.agent_type == "dqn" and epsilon_for_action_selection is not None:
                print_epsilon_str = f"| Epsilon: {epsilon_for_action_selection:.4f}"
            # REINFORCE explores via policy stochasticity, not epsilon traditionally

            loss_str = "Loss: N/A"
            if hasattr(agent, "last_loss") and agent.last_loss is not None:
                try:
                    loss_str = f"Loss: {float(agent.last_loss):.4f}"
                except Exception as e:
                    loss_str = f"Loss: ({e})"

            print(
                f"Epoch: {current_epoch}/{total_learning_epochs} | Game: {games_played_this_run} | "
                f"Score: {current_game_score} | Avg Score: {avg_score:.2f} | "
                f"Lines: {env.cleared_lines} {print_epsilon_str} | {loss_str}"
            )

            # --- Model Saving Logic ---
            if current_game_score > highest_score_this_session:
                highest_score_this_session = current_game_score
                print(
                    f"** New best score for this session: {highest_score_this_session}. Checking against disk. **"
                )
                agent_prefix = get_agent_file_prefix(
                    opt.agent_type
                )  # Generic for DQN/REINFORCE
                disk_best_score = find_best_existing_score(
                    agent_prefix, current_model_base_dir
                )
                if current_game_score > disk_best_score:
                    print(
                        f"Current score {current_game_score} > disk best {opt.agent_type} score {disk_best_score}. Saving."
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
                    agent.save(new_model_path)
                    print(
                        f"Saved {opt.agent_type} model (Score: {current_game_score}) to {new_model_path}"
                    )
                else:
                    print(
                        f"Session best {current_game_score}, but disk best {opt.agent_type} score is {disk_best_score}. Not overwriting disk."
                    )
            # --- End Model Saving ---

            if current_game_score >= EARLY_STOPPING_TARGET_SCORE:
                print(
                    f"\nEarly stopping: Target score {EARLY_STOPPING_TARGET_SCORE} reached."
                )
                training_complete = True

            agent.reset()  # Agent's per-episode/game reset
            current_game_score = 0
            s_t_board_features = env.reset()
            if tetris_config.DEVICE.type == "cuda":
                s_t_board_features = s_t_board_features.cuda()
        else:
            s_t_board_features = s_prime_actual_features

    print("\nTraining finished.")
    if training_complete and current_epoch > 0:
        pass
    elif current_epoch >= total_learning_epochs:
        print(
            f"Completed {current_epoch} learning epochs/episodes as per configuration."
        )

    if highest_score_this_session > -1:
        print(
            f"Highest score achieved in this training session: {highest_score_this_session}"
        )
    elif games_played_this_run > 0:
        print(
            "No new high scores recorded in this session (or learning did not occur often)."
        )
    else:
        print("No games or learning epochs were completed in this session.")


if __name__ == "__main__":
    train()
