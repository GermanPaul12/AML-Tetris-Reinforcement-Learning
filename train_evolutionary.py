# tetris_rl_agents/train_evolutionary.py
import argparse
import os
import random
import numpy as np
import torch

import config as tetris_config
#import test_config as tetris_config
from agents import GeneticAlgorithmController, ESAgent # Import controllers
from src.tetris import Tetris # The game environment

def get_args():
    parser = argparse.ArgumentParser(
        """Train Evolutionary Agents (GA, ES) for Tetris""")
    
    parser.add_argument("--agent_type", type=str, required=True,
                        choices=["genetic", "es"],
                        help="Type of evolutionary agent to train.")
    parser.add_argument("--fps", type=int, default=300, 
                        help="Frames per second for the video. Higher for faster playback.")
    parser.add_argument("--num_generations", type=int, default=None,
                        help="Number of generations to train (overrides agent's config).")
    # Add other relevant CLI args if needed, e.g., population size for quick tests
    
    args = parser.parse_args()
    return args

def setup_seeds():
    torch.manual_seed(tetris_config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tetris_config.SEED)
    np.random.seed(tetris_config.SEED)
    random.seed(tetris_config.SEED)
    print(f"Seeds set to: {tetris_config.SEED}")

def train_genetic_algorithm(env_template: Tetris):
    print("\n--- Training Genetic Algorithm (GA) ---")
    controller = GeneticAlgorithmController(
        state_size=tetris_config.STATE_SIZE,
        seed=tetris_config.SEED
        # Other params will be taken from tetris_config inside the controller
    )
    
    num_generations = tetris_config.GA_N_GENERATIONS # Default from config
    if cli_args.num_generations is not None:
        num_generations = cli_args.num_generations
        print(f"Overriding number of GA generations to: {num_generations}")

    print(f"Starting GA Training for {num_generations} generations...")
    for gen in range(num_generations):
        mean_fitness, max_fitness = controller.evolve_population(env_template)
        print(f"GA Gen {gen + 1}/{num_generations} | Mean Pop Fit: {mean_fitness:.2f} | Max Pop Fit: {max_fitness:.2f} | Best Overall: {controller.best_fitness:.2f}")

        if (gen + 1) % tetris_config.GA_SAVE_EVERY_N_GENERATIONS == 0 or (gen + 1) == num_generations:
            controller.save_best_individual() # Uses default path from config

        # Optional: Early stopping if target fitness is met
        if controller.best_fitness >= tetris_config.SCORE_TARGET: # A general score target
            print(f"\nGA Target Score ({tetris_config.SCORE_TARGET}) reached!")
            break
            
    print("GA Training finished.")
    controller.save_best_individual() # Final save


def train_evolutionary_strategies(env_template: Tetris):
    print("\n--- Training Evolutionary Strategies (ES) ---")
    # ESAgent class itself is the controller
    es_controller = ESAgent(
        state_size=tetris_config.STATE_SIZE,
        seed=tetris_config.SEED
        # Other params from config
    )

    num_generations = tetris_config.ES_N_GENERATIONS # Default from config
    if cli_args.num_generations is not None:
        num_generations = cli_args.num_generations
        print(f"Overriding number of ES generations to: {num_generations}")

    print(f"Starting ES Training for {num_generations} generations...")
    for gen in range(num_generations):
        mean_pop_fit, max_pop_fit, central_fit = es_controller.evolve_step(env_template)
        print(f"ES Gen {gen + 1}/{num_generations} | Mean Pop Fit: {mean_pop_fit:.2f} | Max Pop Fit: {max_pop_fit:.2f} | Central Policy Fit: {central_fit:.2f} | Best Overall: {es_controller.current_best_fitness:.2f}")

        if (gen + 1) % tetris_config.ES_PRINT_EVERY_GENS == 0 or (gen + 1) == num_generations:
            es_controller.save() # Uses default path

        if es_controller.current_best_fitness >= tetris_config.ES_TARGET_GAME_SCORE:
            print(f"\nES Target Score ({tetris_config.ES_TARGET_GAME_SCORE}) reached!")
            break
    
    print("ES Training finished.")
    es_controller.save() # Final save


if __name__ == "__main__":
    cli_args = get_args()
    tetris_config.ensure_model_dir_exists()
    setup_seeds()

    # Create a template environment instance.
    # Specific evaluation environments will be created fresh inside controllers/agents.
    env_template = Tetris(width=tetris_config.GAME_WIDTH,
                          height=tetris_config.GAME_HEIGHT,
                          block_size=tetris_config.GAME_BLOCK_SIZE)

    if cli_args.agent_type == "genetic":
        train_genetic_algorithm(env_template)
    elif cli_args.agent_type == "es":
        train_evolutionary_strategies(env_template)
    else:
        print(f"Error: Agent type '{cli_args.agent_type}' is not supported by this evolutionary training script.")