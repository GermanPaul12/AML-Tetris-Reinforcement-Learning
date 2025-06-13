# tetris_rl_agents/evaluate.py
import os
import csv
import argparse
import traceback
import numpy as np

from helper import *
import config as tetris_config
from agents import AGENT_REGISTRY
from src.tetris import Tetris

from agents.genetic_agent import GeneticAgent as GeneticAgentWrapper

def run_evaluation_game(env: Tetris, agent):
    current_board_features = env.reset()

    if tetris_config.DEVICE.type == "cuda": current_board_features = current_board_features.cuda()
    agent.reset()

    game_score_this_eval_run = 0
    game_over = False
    pieces_played_this_eval_run = 0

    while (not game_over and pieces_played_this_eval_run < tetris_config.MAX_PIECES_PER_EVAL_GAME):
        action_tuple, _ = agent.select_action(current_board_features, env, epsilon_override=0.0)

        render = (tetris_config.RENDER_MODE_EVAL == "human")
        reward_for_step, game_over = env.step(action_tuple, render=render)

        game_score_this_eval_run += reward_for_step
        pieces_played_this_eval_run += 1

        if not game_over:
            current_board_features = env.get_state_properties(env.board)
            if tetris_config.DEVICE.type == "cuda":
                current_board_features = current_board_features.cuda()

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

    if agent_type_to_load == "ppo": actor_path, critic_path = find_latest_or_best_model_path(agent_type_to_load, model_base_dir)
    elif agent_type_to_load != "random": model_load_path = find_latest_or_best_model_path(agent_type_to_load, model_base_dir)

    try:
        if agent_type_to_load == "random":
            print("Initializing Random Agent for evaluation (no model to load).")
        elif agent_type_to_load == "ppo":
            if actor_path and critic_path:
                print(f"Attempting to load PPO Actor: {os.path.basename(actor_path)}")
                print(f"Attempting to load PPO Critic: {os.path.basename(critic_path)}")
                agent_instance.load(actor_path, critic_path)
            else:
                print(f"PPO model files not found in {model_base_dir}. Evaluating untrained PPO.")
        elif model_load_path:
            print(f"Attempting to load model: {os.path.basename(model_load_path)}")
            agent_instance.load(model_load_path)
        elif agent_type_to_load != "random":
            print(f"No model file found for {agent_type_to_load.upper()} in {model_base_dir}. Evaluating untrained agent.")

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
                if (network is not None and hasattr(network, "eval") and callable(network.eval)):
                    network.eval()

        # Special handling for GA if AGENT_REGISTRY points to controller
        if agent_type_to_load == "genetic":
            # If agent_instance is the GAController (from train_evolutionary.py)
            if hasattr(agent_instance, "get_best_policy_network"):
                best_ga_net = agent_instance.get_best_policy_network()
                if best_ga_net:
                    eval_agent = GeneticAgentWrapper(state_size, policy_network_instance=best_ga_net)
                    return eval_agent
                else:
                    print("GA Controller loaded, but no best network found. Evaluating untrained genetic wrapper.")
                    return GeneticAgentWrapper(state_size, seed=tetris_config.SEED + 201)  

        return agent_instance
    except FileNotFoundError as e:
        print(f"  FileNotFoundError during load for {agent_type_to_load}: {e}. Agent will be untrained if possible.")
        return agent_class(state_size=state_size, seed=tetris_config.SEED + 201)
    except Exception as e:
        print(f"  Error loading agent {agent_type_to_load}: {e}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate pre-trained Tetris agents.")
    parser.add_argument("--agent_types", type=str, default="all", help="Comma-separated list of agent types or 'all'.")
    parser.add_argument("--num_eval_games", type=int, default=tetris_config.NUM_EVAL_GAMES, help="Number of games for evaluation.")
    args = parser.parse_args()

    model_base_dir = tetris_config.MODEL_DIR
    eval_master_seed = tetris_config.SEED + 300 # Unique seed for evaluation to avoid conflicts with training seeds

    env = Tetris(
        width=tetris_config.GAME_WIDTH,
        height=tetris_config.GAME_HEIGHT,
        block_size=tetris_config.GAME_BLOCK_SIZE,
    )

    if args.agent_types.lower() == "all":
        agent_types_to_evaluate = [agent.lower() for agent in tetris_config.AGENT_TYPES if agent.lower() in AGENT_REGISTRY]
    else:
        agent_types_to_evaluate = [a.strip().lower() for a in args.agent_types.split(",") if a.strip().lower() in AGENT_REGISTRY]

    evaluation_summary = []

    for agent_type in agent_types_to_evaluate:
        setup_seeds(eval_master_seed)

        agent = load_agent_for_eval(agent_type, tetris_config.STATE_SIZE, model_base_dir)
        if not agent:
            print(f"Skipping evaluation for {agent_type.upper()} due to loading issues.")
            continue

        agent_scores, agent_pieces_played, agent_tetrominoes, agent_lines_cleared = ([], [], [], [])
        print(f"\nEvaluating {agent_type.upper()} for {args.num_eval_games} games...")

        for i_game in range(args.num_eval_games):
            score, pieces, tetrominoes_val, lines_val = run_evaluation_game(env, agent)
            agent_scores.append(score)
            agent_pieces_played.append(pieces)
            agent_tetrominoes.append(tetrominoes_val)
            agent_lines_cleared.append(lines_val)
            print(f"  Game {i_game + 1}/{args.num_eval_games}: Score={score}, Pieces={pieces}, Lines={lines_val}")

        stats = {
            "Agent": agent_type.upper(),
            "Avg Score": np.mean(agent_scores) if agent_scores else 0,
            "Std Score": np.std(agent_scores) if agent_scores else 0,
            "Min Score": np.min(agent_scores) if agent_scores else 0,
            "Max Score": np.max(agent_scores) if agent_scores else 0,
            "Avg Pieces": np.mean(agent_pieces_played) if agent_pieces_played else 0,
            "Avg Tetrominoes": np.mean(agent_tetrominoes) if agent_tetrominoes else 0,
            "Avg Lines Cleared": np.mean(agent_lines_cleared) if agent_lines_cleared else 0,
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

    col_widths = {h: len(h) for h in headers}
    for row in evaluation_summary:
        for h in headers:
            val_str = f"{row[h]:.2f}" if isinstance(row[h], float) else str(row[h])
            col_widths[h] = max(col_widths[h], len(val_str))

    header_line = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    print(header_line)
    print("-" * len(header_line))

    for row in evaluation_summary:
        row_line = " | ".join(f"{(f'{row[h]:.2f}' if isinstance(row[h], float) else str(row[h])):<{col_widths[h]}}" for h in headers)
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
        print(f"\nERROR: Could not write evaluation summary to {tetris_config.EVALUATION_CSV_PATH}: {e}")


if __name__ == "__main__":
    main()
