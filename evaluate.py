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
    current_board_features = env.reset(seed=game_seed)
    if tetris_config.DEVICE.type == 'cuda':
        current_board_features = current_board_features.cuda()
    
    agent.reset() # Reset agent's internal state (e.g. for REINFORCE buffers)
    
    game_score = 0
    game_over = False
    pieces_this_game = 0

    while not game_over and pieces_this_game < tetris_config.MAX_PIECES_PER_EVAL_GAME:
        # For evaluation, use greedy policy (epsilon=0 or deterministic for policy gradients)
        action_tuple, _ = agent.select_action(current_board_features, env, epsilon_override=0.0)
        
        reward, game_over = env.step(action_tuple, render=(tetris_config.RENDER_MODE_EVAL == "human"))
        game_score += reward
        pieces_this_game += 1
        
        if not game_over:
            current_board_features = env.get_state_properties(env.board)
            if tetris_config.DEVICE.type == 'cuda':
                current_board_features = current_board_features.cuda()
        
        if tetris_config.RENDER_MODE_EVAL == "human":
            # env.render() is called inside env.step if render=True
            # Add a small delay if needed for human viewing, though cv2.waitKey(1) in Tetris.render might suffice
            pass

    return game_score, pieces_this_game, env.tetrominoes, env.cleared_lines

def load_agent_for_eval(agent_type_to_load, state_size):
    agent_class = AGENT_REGISTRY.get(agent_type_to_load)
    if not agent_class:
        print(f"Error: Agent type '{agent_type_to_load}' not found.")
        return None

    print(f"Loading agent for evaluation: {agent_type_to_load.upper()}")
    # Seed for agent initialization during eval can be fixed or varied slightly
    agent_instance = agent_class(state_size=state_size, seed=tetris_config.SEED + 200)

    model_load_path = getattr(tetris_config, f"{agent_type_to_load.upper()}_MODEL_PATH", None)

    try:
        if agent_type_to_load == "random":
            print("Initializing Random Agent for evaluation.")
        elif agent_type_to_load == "ppo":
            actor_path = tetris_config.PPO_ACTOR_MODEL_PATH
            critic_path = tetris_config.PPO_CRITIC_MODEL_PATH
            if not os.path.exists(actor_path) or not os.path.exists(critic_path):
                print(f"PPO model (actor or critic) not found. Skipping.")
                return None
            agent_instance.load(actor_path, critic_path)
        elif model_load_path and os.path.exists(model_load_path):
            agent_instance.load(model_load_path)
        elif model_load_path and not os.path.exists(model_load_path):
            print(f"Model for {agent_type_to_load} not found at {model_load_path}. Skipping.")
            return None
        else: # No specific path, could be an agent that doesn't save (like a simple heuristic if added)
            print(f"No specific model path for {agent_type_to_load}. Agent might be untrained or not save.")
            # If it's a trainable agent and no path, it's effectively untrained
            if agent_type_to_load != "random": # Random agent is fine without loading
                 print(f"Warning: Evaluating potentially untrained {agent_type_to_load} agent.")


        # Set to eval mode
        if hasattr(agent_instance, 'qnetwork_local'): agent_instance.qnetwork_local.eval()
        if hasattr(agent_instance, 'qnetwork_target'): agent_instance.qnetwork_target.eval()
        if hasattr(agent_instance, 'policy_network'): agent_instance.policy_network.eval()
        if hasattr(agent_instance, 'central_policy_net'): agent_instance.central_policy_net.eval() # For ES
        if hasattr(agent_instance, 'actor'): agent_instance.actor.eval()
        if hasattr(agent_instance, 'critic'): agent_instance.critic.eval()
        if hasattr(agent_instance, 'network'): agent_instance.network.eval() # For A2C combined

        return agent_instance
    except Exception as e:
        print(f"Error loading agent {agent_type_to_load}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate pre-trained Tetris agents.")
    parser.add_argument(
        "--agent_types",
        type=str,
        default="all",
        help="Comma-separated list of agent types to evaluate (e.g., dqn,ppo,random) or 'all'."
    )
    parser.add_argument(
        "--num_eval_games",
        type=int,
        default=tetris_config.NUM_EVAL_GAMES,
        help=f"Number of games to run for evaluation per agent."
    )
    args = parser.parse_args()

    tetris_config.ensure_model_dir_exists()
    # Use a different seed for evaluation runs than training/testing if desired
    eval_master_seed = tetris_config.SEED + 300 
    setup_seeds(eval_master_seed)

    env = Tetris(width=tetris_config.GAME_WIDTH, height=tetris_config.GAME_HEIGHT,
                 block_size=tetris_config.GAME_BLOCK_SIZE)
    
    state_size = tetris_config.STATE_SIZE

    if args.agent_types.lower() == "all":
        agent_types_to_evaluate = tetris_config.AGENT_TYPES
    else:
        agent_types_to_evaluate = [a.strip() for a in args.agent_types.split(',')]

    evaluation_summary = []

    for agent_type in agent_types_to_evaluate:
        agent = load_agent_for_eval(agent_type, state_size)
        if not agent:
            continue

        agent_scores = []
        agent_pieces = []
        agent_tetrominoes_count = []
        agent_lines_cleared_count = []

        print(f"\nEvaluating {agent_type.upper()} for {args.num_eval_games} games...")
        for i_game in range(args.num_eval_games):
            game_seed = eval_master_seed + i_game + 1 # Unique seed per game
            score, pieces, tetrominoes, lines = run_evaluation_game(env, agent, game_seed)
            agent_scores.append(score)
            agent_pieces.append(pieces)
            agent_tetrominoes_count.append(tetrominoes)
            agent_lines_cleared_count.append(lines)
            print(f"  Game {i_game + 1}/{args.num_eval_games}: Score={score}, Pieces={pieces}, Tetrominoes={tetrominoes}, Lines={lines}")

        stats = {
            "Agent": agent_type.upper(),
            "Avg Score": np.mean(agent_scores) if agent_scores else 0,
            "Std Score": np.std(agent_scores) if agent_scores else 0,
            "Min Score": np.min(agent_scores) if agent_scores else 0,
            "Max Score": np.max(agent_scores) if agent_scores else 0,
            "Avg Pieces": np.mean(agent_pieces) if agent_pieces else 0,
            "Avg Tetrominoes": np.mean(agent_tetrominoes_count) if agent_tetrominoes_count else 0,
            "Avg Lines Cleared": np.mean(agent_lines_cleared_count) if agent_lines_cleared_count else 0,
            "Num Eval Games": len(agent_scores)
        }
        evaluation_summary.append(stats)

    if not evaluation_summary:
        print("\nNo agents were evaluated.")
        return

    print("\n\n--- Evaluation Summary ---")
    headers = ["Agent", "Avg Score", "Std Score", "Min Score", "Max Score", "Avg Pieces", "Avg Tetrominoes", "Avg Lines Cleared", "Num Eval Games"]
    col_widths = {h: len(h) for h in headers}
    for row in evaluation_summary:
        for h in headers:
            val_str = f"{row[h]:.2f}" if isinstance(row[h], float) else str(row[h])
            col_widths[h] = max(col_widths[h], len(val_str))

    header_line = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    print(header_line)
    print("-" * len(header_line))

    for row in evaluation_summary:
        row_line = " | ".join(f"{ (f'{row[h]:.2f}' if isinstance(row[h], float) else str(row[h])) :<{col_widths[h]}}" for h in headers)
        print(row_line)

    try:
        with open(tetris_config.EVALUATION_CSV_PATH, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row_data in evaluation_summary:
                 writer.writerow({h: (f"{row_data[h]:.2f}" if isinstance(row_data[h], float) else row_data[h]) for h in headers})
        print(f"\nEvaluation summary saved to: {tetris_config.EVALUATION_CSV_PATH}")
    except IOError:
        print(f"\nERROR: Could not write evaluation summary to {tetris_config.EVALUATION_CSV_PATH}")

if __name__ == "__main__":
    main()