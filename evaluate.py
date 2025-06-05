# tetris_rl_agents/evaluate.py
import os
import random
import numpy as np
import torch
import csv
import argparse

import config as tetris_config
from agents import AGENT_REGISTRY
from src.tetris import Tetris


def setup_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Evaluation seeds set to: {seed}")


def run_evaluation_game(env: Tetris, agent, game_seed):
    current_board_features = env.reset()
    if tetris_config.DEVICE.type == "cuda":
        current_board_features = current_board_features.cuda()

    agent.reset()

    game_score = 0
    game_over = False
    pieces_this_game = 0

    while not game_over and pieces_this_game < tetris_config.MAX_PIECES_PER_EVAL_GAME:
        action_tuple, _ = agent.select_action(
            current_board_features, env, epsilon_override=0.0
        )

        reward, game_over = env.step(
            action_tuple, render=(tetris_config.RENDER_MODE_EVAL == "human")
        )
        game_score += int(reward)  # Ensure score is int for consistency with training
        pieces_this_game += 1

        if not game_over:
            current_board_features = env.get_state_properties(env.board)
            if tetris_config.DEVICE.type == "cuda":
                current_board_features = current_board_features.cuda()

        if tetris_config.RENDER_MODE_EVAL == "human":
            pass  # cv2.waitKey(1) in Tetris.render handles delay

    return game_score, pieces_this_game, env.tetrominoes, env.cleared_lines


def load_agent_for_eval(agent_type_to_load, state_size):
    agent_class = AGENT_REGISTRY.get(agent_type_to_load)
    if not agent_class:
        print(f"Error: Agent type '{agent_type_to_load}' not found in AGENT_REGISTRY.")
        return None

    print(f"Loading agent for evaluation: {agent_type_to_load.upper()}")
    # Using a fixed seed for agent instantiation during eval for consistency if agent has random init components
    agent_instance = agent_class(state_size=state_size, seed=tetris_config.SEED + 200)

    model_load_path = None
    actor_path = None
    critic_path = None

    # Determine model path(s) based on agent type
    if agent_type_to_load == "ppo":
        actor_path = getattr(tetris_config, "PPO_ACTOR_MODEL_PATH", None)
        critic_path = getattr(tetris_config, "PPO_CRITIC_MODEL_PATH", None)
    else:
        # Construct path variable name, e.g., DQN_MODEL_PATH, REINFORCE_MODEL_PATH
        path_var_name = f"{agent_type_to_load.upper()}_MODEL_PATH"
        model_load_path = getattr(tetris_config, path_var_name, None)

    try:
        if agent_type_to_load == "random":
            print("Initializing Random Agent for evaluation (no model to load).")
        elif agent_type_to_load == "ppo":
            if (
                actor_path
                and critic_path
                and os.path.exists(actor_path)
                and os.path.exists(critic_path)
            ):
                agent_instance.load(actor_path, critic_path)
            else:
                print(
                    f"PPO model (actor path: {actor_path} or critic path: {critic_path}) not found. Evaluating untrained PPO or skipping."
                )
                # Depending on PPOAgent.load behavior if paths are None, it might proceed as untrained.
                # If you want to strictly skip if models not found:
                # return None
        elif model_load_path and os.path.exists(model_load_path):
            agent_instance.load(model_load_path)
        elif model_load_path and not os.path.exists(model_load_path):
            print(
                f"Model for {agent_type_to_load} not found at {model_load_path}. Evaluating untrained agent or skipping."
            )
            # return None # if strictly skip
        elif (
            agent_type_to_load != "random"
        ):  # No specific path defined in config and not random agent
            print(
                f"No MODEL_PATH variable defined in config for {agent_type_to_load.upper()}. Evaluating untrained agent."
            )

        # Set relevant networks to eval mode
        networks_to_eval = [
            "v_network",
            "policy_network",
            "qnetwork_local",
            "qnetwork_target",
            "actor",
            "critic",
            "network",
            "central_policy_net",
        ]
        for net_name in networks_to_eval:
            if hasattr(agent_instance, net_name):
                network = getattr(agent_instance, net_name)
                if network is not None and hasattr(network, "eval"):
                    network.eval()
                    # print(f"  Set {net_name} to eval mode.")

        return agent_instance
    except Exception as e:
        print(f"Error loading or setting up agent {agent_type_to_load}: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate pre-trained Tetris agents.")
    parser.add_argument(
        "--agent_types",
        type=str,
        default="all",
        help="Comma-separated list of agent types (e.g., dqn,ppo,random) or 'all'.",
    )
    parser.add_argument(
        "--num_eval_games",
        type=int,
        default=tetris_config.NUM_EVAL_GAMES,
        help="Number of games to run for evaluation per agent.",
    )
    args = parser.parse_args()

    tetris_config.ensure_model_dir_exists()
    eval_master_seed = tetris_config.SEED + 300
    # setup_seeds(eval_master_seed) # Seed Python, Numpy, PyTorch globally ONCE for the evaluation run

    env = Tetris(
        width=tetris_config.GAME_WIDTH,
        height=tetris_config.GAME_HEIGHT,
        block_size=tetris_config.GAME_BLOCK_SIZE,
    )

    state_size = tetris_config.STATE_SIZE

    if args.agent_types.lower() == "all":
        agent_types_to_evaluate = tetris_config.AGENT_TYPES
    else:
        agent_types_to_evaluate = [
            a.strip().lower() for a in args.agent_types.split(",")
        ]  # Ensure lowercase

    evaluation_summary = []

    for agent_type_raw in agent_types_to_evaluate:
        agent_type = agent_type_raw.lower()  # Normalize to lowercase
        setup_seeds(
            eval_master_seed + sum(ord(c) for c in agent_type)
        )  # Agent-type specific master seed

        agent = load_agent_for_eval(agent_type, state_size)
        if not agent:
            print(
                f"Skipping evaluation for {agent_type.upper()} due to loading issues."
            )
            continue

        agent_scores = []
        agent_pieces = []
        agent_tetrominoes_count = []
        agent_lines_cleared_count = []

        print(f"\nEvaluating {agent_type.upper()} for {args.num_eval_games} games...")
        for i_game in range(args.num_eval_games):
            # game_seed for Tetris env.reset() to vary piece sequences per game
            # Using a combination of master seed and game index for reproducibility of this specific game
            current_game_seed_for_env = (
                eval_master_seed
                + agent_types_to_evaluate.index(agent_type_raw) * args.num_eval_games
                + i_game
            )
            # random.seed(current_game_seed_for_env) # Seed before env.reset() for Tetris bag
            # Note: env.reset() itself calls random.shuffle. Seeding here makes piece sequences per game deterministic.
            # However, this was not in original, Tetris seeds its bag internally.
            # For now, rely on Tetris's own randomization on reset, influenced by initial global seed.

            score, pieces, tetrominoes, lines = run_evaluation_game(
                env, agent, current_game_seed_for_env
            )  # Pass game_seed
            agent_scores.append(score)
            agent_pieces.append(pieces)
            agent_tetrominoes_count.append(tetrominoes)
            agent_lines_cleared_count.append(lines)
            print(
                f"  Game {i_game + 1}/{args.num_eval_games}: Score={score}, Pieces={pieces}, Tetrominoes={tetrominoes}, Lines={lines}"
            )

        stats = {
            "Agent": agent_type.upper(),
            "Avg Score": np.mean(agent_scores) if agent_scores else 0,
            "Std Score": np.std(agent_scores) if agent_scores else 0,
            "Min Score": np.min(agent_scores) if agent_scores else 0,
            "Max Score": np.max(agent_scores) if agent_scores else 0,
            "Avg Pieces": np.mean(agent_pieces) if agent_pieces else 0,
            "Avg Tetrominoes": np.mean(agent_tetrominoes_count)
            if agent_tetrominoes_count
            else 0,
            "Avg Lines Cleared": np.mean(agent_lines_cleared_count)
            if agent_lines_cleared_count
            else 0,
            "Num Eval Games": len(agent_scores),
        }
        evaluation_summary.append(stats)

    if not evaluation_summary:
        print("\nNo agents were successfully evaluated.")
        return

    print("\n\n--- Evaluation Summary ---")
    # ... (rest of the printing and CSV saving logic - no change needed here) ...
    headers = [
        "Agent",
        "Avg Score",
        "Std Score",
        "Min Score",
        "Max Score",
        "Avg Pieces",
        "Avg Tetrominoes",
        "Avg Lines Cleared",
        "Num Eval Games",
    ]
    col_widths = {h: len(h) for h in headers}
    for row in evaluation_summary:
        for h in headers:
            val_str = f"{row[h]:.2f}" if isinstance(row[h], float) else str(row[h])
            col_widths[h] = max(col_widths[h], len(val_str))

    header_line = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    print(header_line)
    print("-" * len(header_line))

    for row in evaluation_summary:
        row_line = " | ".join(
            f"{(f'{row[h]:.2f}' if isinstance(row[h], float) else str(row[h])):<{col_widths[h]}}"
            for h in headers
        )
        print(row_line)

    try:
        with open(tetris_config.EVALUATION_CSV_PATH, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row_data in evaluation_summary:
                writer.writerow(
                    {
                        h: (
                            f"{row_data[h]:.2f}"
                            if isinstance(row_data[h], float)
                            else row_data[h]
                        )
                        for h in headers
                    }
                )
        print(f"\nEvaluation summary saved to: {tetris_config.EVALUATION_CSV_PATH}")
    except IOError:
        print(
            f"\nERROR: Could not write evaluation summary to {tetris_config.EVALUATION_CSV_PATH}"
        )


if __name__ == "__main__":
    main()
