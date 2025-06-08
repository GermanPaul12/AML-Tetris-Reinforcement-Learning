# tetris_rl_agents/test.py
import argparse
import torch
import os
import random
import numpy as np
import re
import imageio  # For GIF creation
import cv2
from PIL import Image
from collections import deque

import config as tetris_config
from agents import AGENT_REGISTRY
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
        "--output_gif_basename",  # Changed from video to GIF
        type=str,
        default="best_tetris_game",
        help="Base name for the output GIF file (score will be appended).",
    )
    # Remove FPS argument since we'll use fixed GIF parameters
    parser.add_argument(
        "--num_games",
        type=int,
        default=5,
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


def _get_rgb_frame_from_env(env: Tetris) -> np.ndarray:
    """
    Replicates the rendering logic from the Tetris class to return an RGB numpy array.
    This is necessary because the original render() method does not return the frame.
    """
    if not env.gameover:
        board_state = env.get_current_board_state()
        img_data = [env.piece_colors[p] for row in board_state for p in row]
    else:
        img_data = [env.piece_colors[p] for row in env.board for p in row]

    img = np.array(img_data).reshape((env.height, env.width, 3)).astype(np.uint8)
    img = Image.fromarray(img, "RGB")
    img = img.resize(
        (env.width * env.block_size, env.height * env.block_size), Image.NEAREST
    )
    img = np.array(img)

    img[[i * env.block_size for i in range(env.height)], :, :] = 0
    img[:, [i * env.block_size for i in range(env.width)], :] = 0
    img = np.concatenate((img, env.extra_board), axis=1)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.putText(
        img_bgr,
        "Score:",
        (env.width * env.block_size + int(env.block_size / 2), env.block_size),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1.0,
        color=env.text_color,
    )
    cv2.putText(
        img_bgr,
        str(env.score),
        (env.width * env.block_size + int(env.block_size / 2), 2 * env.block_size),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1.0,
        color=env.text_color,
    )
    cv2.putText(
        img_bgr,
        "Pieces:",
        (env.width * env.block_size + int(env.block_size / 2), 4 * env.block_size),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1.0,
        color=env.text_color,
    )
    cv2.putText(
        img_bgr,
        str(env.tetrominoes),
        (env.width * env.block_size + int(env.block_size / 2), 5 * env.block_size),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1.0,
        color=env.text_color,
    )
    cv2.putText(
        img_bgr,
        "Lines:",
        (env.width * env.block_size + int(env.block_size / 2), 7 * env.block_size),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1.0,
        color=env.text_color,
    )
    cv2.putText(
        img_bgr,
        str(env.cleared_lines),
        (env.width * env.block_size + int(env.block_size / 2), 8 * env.block_size),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1.0,
        color=env.text_color,
    )

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def play_game(
    env: Tetris,
    agent,
    game_seed,
    record_frames=False,
    max_pieces=10000,
    max_frames_to_keep=1000,  # Add this parameter
):
    """Plays one game of Tetris and returns the score and piece count."""
    random.seed(game_seed)
    np.random.seed(game_seed)
    torch.manual_seed(game_seed)
    if tetris_config.DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(game_seed)

    current_board_features = env.reset()
    if tetris_config.DEVICE.type == "cuda":
        current_board_features = current_board_features.cuda()
    if hasattr(agent, "reset"):
        agent.reset()

    game_score = 0
    game_over = False
    pieces_played = 0
    frames = None

    if record_frames:
        frames = deque(maxlen=max_frames_to_keep)  # Use deque to limit frames
    else:
        frames = None

    while not game_over and pieces_played < max_pieces:
        action_tuple, _ = agent.select_action(
            current_board_features,
            env,
            epsilon_override=0.0,
        )

        if record_frames:
            frame = _get_rgb_frame_from_env(env)
            frames.append(frame)  # Add to deque

        reward, game_over = env.step(action_tuple, render=False)
        game_score += int(reward)
        pieces_played += 1

        if game_over:
            break

        current_board_features = env.get_state_properties(env.board)
        if tetris_config.DEVICE.type == "cuda":
            current_board_features = current_board_features.cuda()

    if record_frames:
        # Capture final frame and convert to list
        frame = _get_rgb_frame_from_env(env)
        frames.append(frame)
        frames = list(frames)  # Convert deque to list

    return game_score, pieces_played, env.tetrominoes, env.cleared_lines, frames


def create_optimized_gif(frames, output_path, target_mb=50):
    """Create a GIF from frames while ensuring it stays under target size"""
    if not frames:
        return False

    print(f"Creating GIF with {len(frames)} frames...")

    try:
        imageio.mimsave(output_path, frames, fps=10, loop=0)
        gif_size = os.path.getsize(output_path) / (1024 * 1024)

        if gif_size <= target_mb:
            print(f"GIF created at {gif_size:.2f} MB (under {target_mb} MB)")
            return True
    except Exception as e:
        print(f"Error creating GIF: {e}")
        return False

    print(f"Initial GIF too large ({gif_size:.2f} MB). Reducing duration...")
    reduction_factors = [0.5, 0.3, 0.2, 0.1]

    for factor in reduction_factors:
        try:
            keep_frames = int(len(frames) * factor)
            reduced_frames = frames[-keep_frames:]

            imageio.mimsave(output_path, reduced_frames, fps=10, loop=0)
            gif_size = os.path.getsize(output_path) / (1024 * 1024)

            if gif_size <= target_mb:
                print(
                    f"Created GIF with last {factor * 100:.0f}% of game ({gif_size:.2f} MB)"
                )
                return True
        except Exception as e:
            print(f"Error during GIF reduction: {e}")
            continue

    print("Failed to create GIF under size limit")
    return False


def test():
    opt = get_args()
    model_base_dir = tetris_config.MODEL_DIR
    tetris_config.ensure_model_dir_exists()

    test_master_seed = tetris_config.SEED + 100
    setup_seeds(test_master_seed)

    env = Tetris(
        width=tetris_config.GAME_WIDTH,
        height=tetris_config.GAME_HEIGHT,
        block_size=tetris_config.GAME_BLOCK_SIZE,
    )
    state_size = tetris_config.STATE_SIZE

    agent_class = AGENT_REGISTRY.get(opt.agent_type)
    if not agent_class:
        print(f"Error: Agent type '{opt.agent_type}' not found in AGENT_REGISTRY.")
        return

    print(f"\n--- Testing Agent: {opt.agent_type.upper()} ---")
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
        loaded_successfully = True
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

    print(f"\nPlaying {opt.num_games} games to identify the best run...")
    best_score_found = -float("inf")
    seed_for_best_game = None
    all_test_scores = []

    # First pass: find best game without recording
    for i_game in range(opt.num_games):
        current_game_seed = test_master_seed + 1000 + i_game
        score, pieces, _, _, _ = play_game(
            env,
            agent,
            current_game_seed,
            record_frames=False,  # No recording in first pass
            max_pieces=tetris_config.MAX_PIECES_PER_EVAL_GAME,
        )
        all_test_scores.append(score)
        print(f"  Game {i_game + 1}/{opt.num_games}: Score={score}, Pieces={pieces}")

        if score > best_score_found:
            best_score_found = score
            seed_for_best_game = current_game_seed
            print(f"    New best score: {best_score_found} (seed: {seed_for_best_game})")

    if seed_for_best_game is None:
        print("\nNo games were successfully played. Cannot create GIF.")
        return

    # Replay best game with recording
    print(f"\nReplaying best game (Seed: {seed_for_best_game}) to record GIF...")
    score, pieces, _, _, best_game_frames = play_game(
        env,
        agent,
        seed_for_best_game,
        record_frames=True,
        max_pieces=tetris_config.MAX_PIECES_PER_EVAL_GAME,
        max_frames_to_keep=1000  # Only keep last 1000 frames
    )

    gif_filename = (
        f"{opt.output_gif_basename}_score_{best_score_found}_{opt.agent_type}.gif"
    )
    gif_path = os.path.join(tetris_config.MODEL_DIR, "gifs", gif_filename)
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    print(f"\nCreating GIF for best game (Score: {best_score_found})...")
    success = create_optimized_gif(best_game_frames, gif_path, target_mb=50)

    if success:
        print(f"GIF created successfully: {gif_path}")
    else:
        print(f"Failed to create GIF: {gif_path}")

    print(
        f"\nTesting finished. Best game (Score: {best_score_found}) GIF saved to: {gif_path}"
    )
    if all_test_scores:
        print(f"  Scores from all {opt.num_games} preliminary runs: {all_test_scores}")
        print(
            f"  Average score over {opt.num_games} runs: {np.mean(all_test_scores):.2f}"
        )


if __name__ == "__main__":
    test()
