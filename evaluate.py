# tetris_rl_agents/evaluate.py
import os
import random
import numpy as np
import torch
import csv
import argparse
import re
import sys

import config as tetris_config
from agents import AGENT_REGISTRY
from src.tetris import Tetris


# --- Helper Functions (Copied from train_evolutionary.py, should be in a shared util) ---
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

    if not os.path.isdir(model_dir):
        print(
            f"Warning: Model directory {model_dir} does not exist for find_latest_or_best_model_path."
        )
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
    found_files_with_scores = []  # list of (score, path, mtime)
    found_files_no_scores = []  # list of (path, mtime)

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
    elif found_files_no_scores:  # Fallback to most recent if no scored files
        found_files_no_scores.sort(key=lambda x: x[1], reverse=True)
        return found_files_no_scores[0][0]

    return None


# --- End Helper Functions ---


def setup_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # print(f"Evaluation seeds set to: {seed}") # Optional: for debugging


def run_evaluation_game(env: Tetris, agent, game_seed_for_env_reset_unused):
    # env.reset() will use its own internal randomization or previously set global seed
    initial_total_lines = env.cleared_lines  # Store cumulative lines before this game
    initial_total_tetrominoes = env.tetrominoes  # Store cumulative tetrominoes

    current_board_features = env.reset()  # This resets env.score, env.tetrominoes_this_game, env.lines_cleared_this_game IF THEY EXIST
    # Since they don't, env.score, env.cleared_lines, env.tetrominoes are reset by env.reset()

    # If Tetris.reset() doesn't reset its own score, lines, tetrominoes to 0, we need to adjust.
    # Assuming Tetris.reset() correctly resets these for a new game scenario.
    # If not, then `game_score = env.score - initial_score_for_this_instance` would be needed.
    # For now, assume Tetris.reset() correctly sets env.score to 0 for a new game.

    if tetris_config.DEVICE.type == "cuda":
        current_board_features = current_board_features.cuda()
    if hasattr(agent, "reset"):
        agent.reset()

    game_score_this_eval_run = 0  # This will be the true score for THIS evaluation game
    game_over = False
    pieces_played_this_eval_run = 0

    while (
        not game_over
        and pieces_played_this_eval_run < tetris_config.MAX_PIECES_PER_EVAL_GAME
    ):
        action_tuple, _ = agent.select_action(
            current_board_features, env, epsilon_override=0.0
        )

        should_render_this_step = tetris_config.RENDER_MODE_EVAL == "human"
        # The reward from env.step is the incremental score for that step
        reward_for_step, game_over = env.step(
            action_tuple, render=should_render_this_step
        )

        game_score_this_eval_run += int(reward_for_step)
        pieces_played_this_eval_run += 1

        if not game_over:
            current_board_features = env.get_state_properties(env.board)
            if tetris_config.DEVICE.type == "cuda":
                current_board_features = current_board_features.cuda()

        if should_render_this_step and hasattr(
            env, "render"
        ):  # Check if render is callable if it uses OpenCV
            if (
                "cv2" in sys.modules
            ):  # Only if cv2 is imported by tetris.py for rendering
                sys.modules["cv2"].waitKey(1)

    return env.score, pieces_played_this_eval_run, env.tetrominoes, env.cleared_lines


def load_agent_for_eval(agent_type_to_load, state_size, model_base_dir):
    agent_class = AGENT_REGISTRY.get(agent_type_to_load)
    if not agent_class:
        print(f"Error: Agent type '{agent_type_to_load}' not found in AGENT_REGISTRY.")
        return None

    print(f"Loading agent for evaluation: {agent_type_to_load.upper()}")
    agent_instance = agent_class(state_size=state_size, seed=tetris_config.SEED + 200)

    actor_path, critic_path = None, None
    model_load_path = None

    if agent_type_to_load == "ppo":
        actor_path, critic_path = find_latest_or_best_model_path(
            agent_type_to_load, model_base_dir
        )
    elif agent_type_to_load != "random":
        model_load_path = find_latest_or_best_model_path(
            agent_type_to_load, model_base_dir
        )

    try:
        if agent_type_to_load == "random":
            print("Initializing Random Agent for evaluation (no model to load).")
        elif agent_type_to_load == "ppo":
            if actor_path and critic_path:
                print(f"  Attempting to load PPO Actor: {os.path.basename(actor_path)}")
                print(
                    f"  Attempting to load PPO Critic: {os.path.basename(critic_path)}"
                )
                agent_instance.load(actor_path, critic_path)
            else:
                print(
                    f"  PPO model files not found in {model_base_dir}. Evaluating untrained PPO."
                )
        elif model_load_path:
            print(f"  Attempting to load model: {os.path.basename(model_load_path)}")
            agent_instance.load(model_load_path)
        elif agent_type_to_load != "random":
            print(
                f"  No model file found for {agent_type_to_load.upper()} in {model_base_dir}. Evaluating untrained agent."
            )

        # Set networks to eval mode
        networks_to_eval = [
            "policy_network",
            "qnetwork_local",
            "qnetwork_target",
            "actor",
            "critic",
            "network",
            "central_policy_net",
            "best_individual_network",
        ]
        for net_name in networks_to_eval:
            if hasattr(agent_instance, net_name):
                network = getattr(agent_instance, net_name)
                if (
                    network is not None
                    and hasattr(network, "eval")
                    and callable(network.eval)
                ):
                    network.eval()

        # Special handling for GA if AGENT_REGISTRY points to controller
        if agent_type_to_load == "genetic":
            # If agent_instance is the GAController (from train_evolutionary.py)
            if hasattr(agent_instance, "get_best_policy_network"):
                best_ga_net = agent_instance.get_best_policy_network()
                if best_ga_net:
                    # Create the GeneticAgent wrapper for evaluation
                    from agents.genetic_agent import GeneticAgent as GeneticAgentWrapper

                    eval_agent = GeneticAgentWrapper(
                        state_size, policy_network_instance=best_ga_net
                    )
                    return eval_agent
                else:
                    print(
                        "  GA Controller loaded, but no best network found. Evaluating untrained genetic wrapper."
                    )
                    from agents.genetic_agent import GeneticAgent as GeneticAgentWrapper

                    return GeneticAgentWrapper(
                        state_size, seed=tetris_config.SEED + 201
                    )  # Untrained wrapper
            # If agent_instance is already the GeneticAgent wrapper, its load should set policy_network
            elif hasattr(agent_instance, "policy_network"):
                pass  # Assume its load method set up the policy_network

        return agent_instance
    except FileNotFoundError as e:
        print(
            f"  FileNotFoundError during load for {agent_type_to_load}: {e}. Agent will be untrained if possible."
        )
        return agent_class(
            state_size=state_size, seed=tetris_config.SEED + 201
        )  # Return fresh instance
    except Exception as e:
        print(f"  Error loading agent {agent_type_to_load}: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate pre-trained Tetris agents.")
    parser.add_argument(
        "--agent_types",
        type=str,
        default="all",
        help="Comma-separated list of agent types or 'all'.",
    )
    parser.add_argument(
        "--num_eval_games",
        type=int,
        default=tetris_config.NUM_EVAL_GAMES,
        help="Number of games for evaluation.",
    )
    args = parser.parse_args()

    model_base_dir = tetris_config.MODEL_DIR
    tetris_config.ensure_model_dir_exists()

    eval_master_seed = tetris_config.SEED + 300

    env = Tetris(
        width=tetris_config.GAME_WIDTH,
        height=tetris_config.GAME_HEIGHT,
        block_size=tetris_config.GAME_BLOCK_SIZE,
    )
    state_size = tetris_config.STATE_SIZE

    if args.agent_types.lower() == "all":
        # Ensure AGENT_REGISTRY keys are lowercase for consistent matching
        agent_types_to_evaluate = [
            agt for agt in tetris_config.AGENT_TYPES if agt.lower() in AGENT_REGISTRY
        ]
    else:
        agent_types_to_evaluate = [
            a.strip().lower()
            for a in args.agent_types.split(",")
            if a.strip().lower() in AGENT_REGISTRY
        ]

    evaluation_summary = []

    for agent_type_raw in agent_types_to_evaluate:
        agent_type = agent_type_raw.lower()

        current_agent_master_seed = eval_master_seed + sum(ord(c) for c in agent_type)
        setup_seeds(current_agent_master_seed)

        agent = load_agent_for_eval(agent_type, state_size, model_base_dir)
        if not agent:
            print(
                f"Skipping evaluation for {agent_type.upper()} due to loading issues."
            )
            continue

        agent_scores, agent_pieces_played, agent_tetrominoes, agent_lines_cleared = (
            [],
            [],
            [],
            [],
        )  # Use distinct names
        print(f"\nEvaluating {agent_type.upper()} for {args.num_eval_games} games...")

        for i_game in range(args.num_eval_games):
            # The env.reset() is called inside run_evaluation_game.
            # Seeding for piece sequence consistency per game (if Tetris uses random.shuffle on a fixed bag):
            # random.seed(current_agent_master_seed + i_game + 1) # +1 to differ from agent init seed

            # run_evaluation_game returns: score, pieces_in_this_game, tetrominoes_in_this_game, lines_in_this_game
            score, pieces, tetrominoes_val, lines_val = run_evaluation_game(
                env, agent, 0
            )
            agent_scores.append(score)
            agent_pieces_played.append(
                pieces
            )  # Use the pieces played in this specific game
            agent_tetrominoes.append(tetrominoes_val)
            agent_lines_cleared.append(lines_val)
            print(
                f"  Game {i_game + 1}/{args.num_eval_games}: Score={score}, Pieces={pieces}, Lines={lines_val}"
            )

        stats = {
            "Agent": agent_type.upper(),
            "Avg Score": np.mean(agent_scores) if agent_scores else 0,
            "Std Score": np.std(agent_scores) if agent_scores else 0,
            "Min Score": np.min(agent_scores) if agent_scores else 0,
            "Max Score": np.max(agent_scores) if agent_scores else 0,
            "Avg Pieces": np.mean(agent_pieces_played)
            if agent_pieces_played
            else 0,  # Use the correct list
            "Avg Tetrominoes": np.mean(agent_tetrominoes) if agent_tetrominoes else 0,
            "Avg Lines Cleared": np.mean(agent_lines_cleared)
            if agent_lines_cleared
            else 0,
            "Num Eval Games": len(agent_scores),
        }
        evaluation_summary.append(stats)

    if not evaluation_summary:
        print("\nNo agents were successfully evaluated.")
        return

    print("\n\n--- Evaluation Summary ---")
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
    # ... (rest of printing and CSV saving logic - no change needed here) ...
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
        eval_dir = os.path.dirname(tetris_config.EVALUATION_CSV_PATH)
        if eval_dir:
            os.makedirs(eval_dir, exist_ok=True)

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
    except IOError as e:
        print(
            f"\nERROR: Could not write evaluation summary to {tetris_config.EVALUATION_CSV_PATH}: {e}"
        )


if __name__ == "__main__":
    main()
