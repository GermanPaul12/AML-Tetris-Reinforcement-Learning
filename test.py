# tetris_rl_agents/test.py
import argparse
import torch
import cv2  # For video writing
import os
import random
import numpy as np

import config as tetris_config
from agents import AGENT_REGISTRY
from src.tetris import Tetris


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
        "--output_video",
        type=str,
        default="output_tetris.mp4",
        help="Name of the output video file.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the video. Higher for faster playback.",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=tetris_config.NUM_TEST_RUNS_GIF,
        help="Number of games to play and record in the video.",
    )

    args = parser.parse_args()
    return args


def test():
    opt = get_args()
    tetris_config.ensure_model_dir_exists()

    # Use a distinct seed for testing to ensure reproducibility of test runs
    test_seed = tetris_config.SEED + 100
    if torch.cuda.is_available():
        torch.cuda.manual_seed(test_seed)
    else:
        torch.manual_seed(test_seed)
    random.seed(test_seed)  # For Python's random, used by Tetris bag
    np.random.seed(test_seed)  # If any part of agent or env uses numpy random

    env = Tetris(
        width=tetris_config.GAME_WIDTH,
        height=tetris_config.GAME_HEIGHT,
        block_size=tetris_config.GAME_BLOCK_SIZE,
    )

    # --- Agent Initialization and Loading ---
    agent_class = AGENT_REGISTRY.get(opt.agent_type)
    if not agent_class:
        print(f"Error: Agent type '{opt.agent_type}' not found in AGENT_REGISTRY.")
        return

    print(f"\n--- Testing Agent: {opt.agent_type.upper()} ---")
    # Agent is instantiated with the test_seed for consistent initialization if it has random components
    agent = agent_class(state_size=tetris_config.STATE_SIZE, seed=test_seed)

    model_load_path = None
    actor_path = None
    critic_path = None

    if opt.agent_type == "ppo":
        actor_path = getattr(tetris_config, "PPO_ACTOR_MODEL_PATH", None)
        critic_path = getattr(tetris_config, "PPO_CRITIC_MODEL_PATH", None)
    else:
        path_var_name = f"{opt.agent_type.upper()}_MODEL_PATH"
        model_load_path = getattr(tetris_config, path_var_name, None)

    if opt.agent_type == "random":
        print("Testing Random Agent. No model to load.")
    elif opt.agent_type == "ppo":
        if (
            actor_path
            and critic_path
            and os.path.exists(actor_path)
            and os.path.exists(critic_path)
        ):
            agent.load(actor_path, critic_path)
            print(
                f"Loaded PPO models: Actor from {actor_path}, Critic from {critic_path}"
            )
        else:
            print(
                f"ERROR: PPO model (actor path: {actor_path} or critic path: {critic_path}) not found. Cannot test."
            )
            return
    elif model_load_path:
        if not os.path.exists(model_load_path):
            print(
                f"ERROR: Model for {opt.agent_type} not found at {model_load_path}. Train first or check path."
            )
            return
        agent.load(model_load_path)
        print(f"Loaded {opt.agent_type} model from {model_load_path}")
    elif opt.agent_type != "random":  # Not random and no specific path found/defined
        print(
            f"Warning: No specific model path for {opt.agent_type} in config or path does not exist. Agent will be effectively untrained for testing."
        )
        # Allow testing untrained (e.g. initial random policy of a complex agent)

    # --- Set agent to evaluation mode ---
    # Generic way to set networks to eval mode
    networks_to_set_eval = [
        "v_network",
        "policy_network",
        "qnetwork_local",
        "qnetwork_target",
        "actor",
        "critic",
        "network",
        "central_policy_net",
    ]
    agent_has_evaluable_net = False
    for net_name in networks_to_set_eval:
        if hasattr(agent, net_name):
            network_component = getattr(agent, net_name)
            if network_component is not None and hasattr(network_component, "eval"):
                network_component.eval()
                print(f"  Set agent's '{net_name}' to evaluation mode.")
                agent_has_evaluable_net = True

    if not agent_has_evaluable_net and opt.agent_type != "random":
        print(
            f"  Note: Agent '{opt.agent_type}' does not seem to have standard named networks for eval mode setting, or they are None."
        )

    # Video writer setup
    render_width = (
        tetris_config.GAME_WIDTH * tetris_config.GAME_BLOCK_SIZE
        + tetris_config.GAME_WIDTH * tetris_config.GAME_BLOCK_SIZE // 2
    )
    render_height = tetris_config.GAME_HEIGHT * tetris_config.GAME_BLOCK_SIZE

    video_path = os.path.join(tetris_config.MODEL_DIR, opt.output_video)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # For .mp4
    out_video = cv2.VideoWriter(
        video_path, fourcc, opt.fps, (render_width, render_height)
    )
    if not out_video.isOpened():
        print(
            f"Error: Could not open video writer for path {video_path}. Check OpenCV/codec installation."
        )
        return

    total_scores = []
    print(
        f"\nStarting testing for {opt.num_games} game(s)... Video will be saved to {video_path}"
    )

    for i_game in range(opt.num_games):
        print(f"  Playing Test Game: {i_game + 1}/{opt.num_games}")

        # Reset environment for each game. Tetris env handles its own piece sequence randomization on reset.
        current_board_features = env.reset()
        if tetris_config.DEVICE.type == "cuda":
            current_board_features = current_board_features.cuda()

        agent.reset()  # Reset agent's per-game state (e.g., REINFORCE buffers, if any)

        game_score = 0
        game_over = False

        # Using MAX_PIECES_PER_EVAL_GAME as a safety limit for test games too
        max_test_pieces = tetris_config.MAX_PIECES_PER_EVAL_GAME

        for piece_num in range(max_test_pieces):
            # Use epsilon_override=0.0 for greedy/deterministic action selection
            action_tuple, _ = agent.select_action(
                current_board_features, env, epsilon_override=0.0
            )

            # env.step renders if render=True AND video object is passed
            reward, game_over = env.step(action_tuple, render=True, video=out_video)
            game_score += int(reward)  # Ensure score is int

            if game_over:
                # Render final game_over state once more to ensure it's in the video
                env.render(video=out_video)
                break  # Exit piece loop for this game

            current_board_features = env.get_state_properties(env.board)
            if tetris_config.DEVICE.type == "cuda":
                current_board_features = current_board_features.cuda()

        print(
            f"    Game {i_game + 1} finished. Score: {game_score}, Tetrominoes: {env.tetrominoes}, Lines: {env.cleared_lines}"
        )
        total_scores.append(game_score)

    out_video.release()
    cv2.destroyAllWindows()  # Close any OpenCV windows opened by env.render if not handled internally
    print(f"\nTesting finished. Video saved to: {video_path}")
    if total_scores:
        print(
            f"  Average score over {opt.num_games} games: {np.mean(total_scores):.2f}"
        )
        print(
            f"  Max score: {np.max(total_scores):.2f}, Min score: {np.min(total_scores):.2f}"
        )
    else:
        print("  No games were completed to calculate scores.")


if __name__ == "__main__":
    test()
