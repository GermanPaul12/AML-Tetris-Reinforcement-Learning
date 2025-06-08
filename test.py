# tetris_rl_agents/test.py
import argparse
import torch
import cv2
import os
import random
import numpy as np
import re
import sys

import config as tetris_config
from agents import (
    AGENT_REGISTRY,
)  # Ensure this is correctly populated with your Tetris agents
from src.tetris import Tetris


# --- Helper Functions (Ideally from a shared utils.py) ---
def get_agent_file_prefix(agent_type_str, is_actor=False, is_critic=False):
    processed_agent_type = agent_type_str.replace("_", "-")
    if agent_type_str == "ppo":
        if is_actor:
            return "ppo-actor"
        elif is_critic:
            return "ppo-critic"
        else:
            return "ppo-model"
    if agent_type_str == "genetic":
        return "genetic"
    if agent_type_str == "es":
        return "es"
    return processed_agent_type


def parse_score_from_filename(filename_basename, expected_prefix):
    pattern_score = re.compile(f"^{re.escape(expected_prefix)}_score_(\\d+)\\.pth$")
    match = pattern_score.match(filename_basename)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def find_latest_or_best_model_path(agent_type_str, model_dir):
    best_score = -1
    best_model_path = None

    if not os.path.isdir(model_dir):
        print(f"Warning: Model directory {model_dir} does not exist.")
        return (None, None) if agent_type_str == "ppo" else None

    if agent_type_str == "ppo":
        actor_prefix = get_agent_file_prefix(agent_type_str, is_actor=True)
        critic_prefix = get_agent_file_prefix(agent_type_str, is_critic=True)
        potential_ppo_models = {}
        for filename in os.listdir(model_dir):
            if filename.startswith(actor_prefix) and filename.endswith(".pth"):
                score_str = filename.replace(f"{actor_prefix}_score_", "").replace(
                    ".pth", ""
                )
                try:
                    score = int(score_str)
                    critic_filename = f"{critic_prefix}_score_{score}.pth"
                    actor_full_path = os.path.join(model_dir, filename)
                    critic_full_path = os.path.join(model_dir, critic_filename)
                    if os.path.exists(critic_full_path):
                        mtime = os.path.getmtime(actor_full_path)
                        if (
                            score not in potential_ppo_models
                            or mtime > potential_ppo_models[score][2]
                        ):
                            potential_ppo_models[score] = (
                                actor_full_path,
                                critic_full_path,
                                mtime,
                            )
                except ValueError:
                    continue
        if potential_ppo_models:
            best_score = max(potential_ppo_models.keys())
            return potential_ppo_models[best_score][:2]
        return None, None

    agent_prefix = get_agent_file_prefix(agent_type_str)
    found_files_with_scores = []
    found_files_no_scores = []
    for filename in os.listdir(model_dir):
        if filename.startswith(agent_prefix) and filename.endswith(".pth"):
            score = parse_score_from_filename(filename, agent_prefix)
            file_path = os.path.join(model_dir, filename)
            mtime = os.path.getmtime(file_path)
            if score is not None:
                found_files_with_scores.append((score, file_path, mtime))
            else:
                found_files_no_scores.append((file_path, mtime))
    if found_files_with_scores:
        found_files_with_scores.sort(key=lambda x: (x[0], x[2]), reverse=True)
        return found_files_with_scores[0][1]
    elif found_files_no_scores:
        found_files_no_scores.sort(key=lambda x: x[1], reverse=True)
        return found_files_no_scores[0][0]
    return None


# --- End Helper Functions ---


def get_args():
    parser = argparse.ArgumentParser(
        """Test pre-trained Reinforcement Learning Agents for Tetris"""
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="dqn",
        choices=list(tetris_config.AGENT_TYPES),
        help="Type of agent to test.",
    )
    parser.add_argument(
        "--output_video_basename",
        type=str,
        default="best_tetris_game",
        help="Base name for the output video file (score will be appended).",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for the video."
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=5,  # Default to 5 games to find a best one
        help="Number of games to play to find the best one to record.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)  # This affects Tetris piece sequence if env uses random.shuffle


def play_game(
    env: Tetris,
    agent,
    game_seed,
    render_for_video=False,
    video_writer=None,
    max_pieces=10000,
):
    """Plays one game of Tetris and returns the score and piece count."""
    # Seed for this specific game run to ensure reproducibility if needed
    # Note: Tetris class internal `random.shuffle(self.bag)` uses Python's global `random`
    random.seed(game_seed)
    np.random.seed(game_seed)  # If agent uses numpy random
    torch.manual_seed(game_seed)  # If agent uses torch random
    if tetris_config.DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(game_seed)

    current_board_features = (
        env.reset()
    )  # Tetris.reset() will use the current random state
    if tetris_config.DEVICE.type == "cuda":
        current_board_features = current_board_features.cuda()
    if hasattr(agent, "reset"):
        agent.reset()

    game_score = 0
    game_over = False
    pieces_played = 0

    while not game_over and pieces_played < max_pieces:
        action_tuple, _ = agent.select_action(
            current_board_features,
            env,
            epsilon_override=0.0,  # Always greedy for testing
        )

        # For video recording, pass the video_writer object to env.step's render
        reward, game_over = env.step(
            action_tuple,
            render=render_for_video,  # Only render if making video
            video=video_writer if render_for_video else None,
        )
        game_score += int(reward)
        pieces_played += 1

        if game_over:
            if render_for_video and video_writer:
                env.render(video=video_writer)  # Ensure final frame is written
            break

        current_board_features = env.get_state_properties(env.board)
        if tetris_config.DEVICE.type == "cuda":
            current_board_features = current_board_features.cuda()

    return game_score, pieces_played, env.tetrominoes, env.cleared_lines


def test():
    opt = get_args()
    model_base_dir = tetris_config.MODEL_DIR
    tetris_config.ensure_model_dir_exists()

    # Master seed for the entire test script run
    test_master_seed = tetris_config.SEED + 100
    setup_seeds(test_master_seed)  # Set global seeds once

    env = Tetris(
        width=tetris_config.GAME_WIDTH,
        height=tetris_config.GAME_HEIGHT,
        block_size=tetris_config.GAME_BLOCK_SIZE,
    )
    state_size = tetris_config.STATE_SIZE

    # --- Agent Initialization and Loading ---
    agent_class = AGENT_REGISTRY.get(opt.agent_type)
    if not agent_class:
        print(f"Error: Agent type '{opt.agent_type}' not found in AGENT_REGISTRY.")
        return

    print(f"\n--- Testing Agent: {opt.agent_type.upper()} ---")
    # Seed for agent instantiation itself (if its __init__ has random parts)
    agent_init_seed = test_master_seed + sum(ord(c) for c in opt.agent_type)
    agent = agent_class(state_size=state_size, seed=agent_init_seed)

    actor_path, critic_path, model_load_path = None, None, None
    if opt.agent_type == "ppo":
        actor_path, critic_path = find_latest_or_best_model_path(
            opt.agent_type, model_base_dir
        )
    elif opt.agent_type != "random":
        model_load_path = find_latest_or_best_model_path(opt.agent_type, model_base_dir)

    loaded_successfully = False
    if opt.agent_type == "random":
        print("Testing Random Agent. No model to load.")
        loaded_successfully = True  # Considered "loaded" for testing
    elif opt.agent_type == "ppo":
        if actor_path and critic_path:
            try:
                agent.load(actor_path, critic_path)
                print(
                    f"Loaded PPO models: Actor from {os.path.basename(actor_path)}, Critic from {os.path.basename(critic_path)}"
                )
                loaded_successfully = True
            except Exception as e:
                print(f"ERROR loading PPO models: {e}")
        else:
            print(f"ERROR: PPO model files not found in {model_base_dir}. Cannot test.")
            return
    elif model_load_path:
        try:
            agent.load(model_load_path)
            print(
                f"Loaded {opt.agent_type} model from {os.path.basename(model_load_path)}"
            )
            loaded_successfully = True
        except Exception as e:
            print(f"ERROR loading model {os.path.basename(model_load_path)}: {e}")

    if not loaded_successfully and opt.agent_type != "random":
        print(
            f"Warning: Could not load a trained model for {opt.agent_type}. Agent will perform as if untrained."
        )
        # Allow testing of an "untrained" (freshly initialized) agent if model load fails or no model exists

    # Set agent networks to evaluation mode
    networks_to_set_eval = [
        "policy_network",
        "qnetwork_local",
        "qnetwork_target",
        "actor",
        "critic",
        "network",
        "central_policy_net",
        "best_individual_network",
    ]
    for net_name in networks_to_set_eval:
        if hasattr(agent, net_name):
            network_component = getattr(agent, net_name)
            if (
                network_component is not None
                and hasattr(network_component, "eval")
                and callable(network_component.eval)
            ):
                network_component.eval()

    # --- Phase 1: Play games to find the best score and its seed ---
    print(f"\nPlaying {opt.num_games} games to identify the best run...")
    best_score_found = -float("inf")
    seed_for_best_game = None
    all_test_scores = []

    for i_game in range(opt.num_games):
        # Each game gets a unique, deterministic seed based on the master test seed and game index
        current_game_seed = test_master_seed + 1000 + i_game

        # Play game without video recording
        score, pieces, _, _ = play_game(
            env,
            agent,
            current_game_seed,
            render_for_video=False,
            video_writer=None,
            max_pieces=tetris_config.MAX_PIECES_PER_EVAL_GAME,
        )
        all_test_scores.append(score)
        print(
            f"  Preliminary Game {i_game + 1}/{opt.num_games}: Score={score}, Pieces={pieces}"
        )
        if score > best_score_found:
            best_score_found = score
            seed_for_best_game = current_game_seed
            print(
                f"    New best score in this test batch: {best_score_found} (seed: {seed_for_best_game})"
            )

    if seed_for_best_game is None:
        print(
            "\nNo games were successfully played, or no positive scores. Cannot record best game."
        )
        if all_test_scores:
            print(f"  Scores from preliminary runs: {all_test_scores}")
        return

    print(
        f"\nBest score found: {best_score_found}. Replaying and recording game with seed: {seed_for_best_game}."
    )

    # --- Phase 2: Replay the best game and record it ---
    video_filename = (
        f"{opt.output_video_basename}_score_{best_score_found}_{opt.agent_type}.mp4"
    )
    video_path = os.path.join(
        tetris_config.MODEL_DIR, "test_videos", video_filename
    )  # Save videos in a subfolder
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    render_width = (
        tetris_config.GAME_WIDTH * tetris_config.GAME_BLOCK_SIZE
        + tetris_config.GAME_WIDTH * tetris_config.GAME_BLOCK_SIZE // 2
    )
    render_height = tetris_config.GAME_HEIGHT * tetris_config.GAME_BLOCK_SIZE
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(
        video_path, fourcc, opt.fps, (render_width, render_height)
    )

    if not out_video.isOpened():
        print(f"Error: Could not open video writer for path {video_path}.")
        return

    # Re-play the best game with its specific seed and record it
    final_score, pieces, _, _ = play_game(
        env,
        agent,
        seed_for_best_game,
        render_for_video=True,
        video_writer=out_video,
        max_pieces=tetris_config.MAX_PIECES_PER_EVAL_GAME,
    )

    out_video.release()
    if "cv2" in sys.modules:  # Only if cv2 was used for rendering
        cv2.destroyAllWindows()

    print(
        f"\nTesting finished. Best game (Score: {final_score}) video saved to: {video_path}"
    )
    if all_test_scores:
        print(f"  Scores from all {opt.num_games} preliminary runs: {all_test_scores}")
        print(
            f"  Average score over {opt.num_games} runs: {np.mean(all_test_scores):.2f}"
        )


if __name__ == "__main__":
    test()
