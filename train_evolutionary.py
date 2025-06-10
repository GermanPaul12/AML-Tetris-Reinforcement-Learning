# AML-TETRIS-RL/train_evolutionary.py

import os
import argparse
import config as tetris_config

from agents import GeneticAlgorithmController, ESAgent
from helper import *
from src.tetris import Tetris

def get_args() -> argparse.Namespace:
    """Parses command line arguments for training evolutionary agents."""
    parser = argparse.ArgumentParser("""Train Evolutionary Agents (GA, ES) for Tetris""")
    parser.add_argument("--agent_type", type=str, required=True, choices=["genetic", "es"], help="Type of evolutionary agent to train.")
    parser.add_argument("--fps", type=int, default=300, help="Frames per second for video (if generated, not used in training).")
    parser.add_argument("--num_generations", type=int, default=None, help="Number of generations to train (overrides agent's config).")
    args = parser.parse_args()
    return args

###########################################################
# Main Training Function for Evolutionary Agents (GA, ES) #
###########################################################

def train_genetic_algorithm(env_template: Tetris, cli_args_ns:argparse.Namespace, model_base_dir: str) -> None:
    print("\n--- Training Genetic Algorithm (GA) ---")
    ga_controller = GeneticAlgorithmController(state_size=tetris_config.STATE_SIZE, seed=tetris_config.SEED)

    num_generations = tetris_config.GA_N_GENERATIONS
    if cli_args_ns.num_generations is not None:
        num_generations = cli_args_ns.num_generations
        print(f"Overriding number of GA generations to: {num_generations}")

    agent_prefix = get_agent_file_prefix(cli_args_ns.agent_type)
    highest_overall_fitness_on_disk = find_best_existing_score(agent_prefix, model_base_dir)
    print(f"Initial best GA score on disk: {highest_overall_fitness_on_disk}")

    print(f"Starting GA Training for {num_generations} generations...")
    for gen in range(num_generations):
        mean_fitness, max_fitness_this_gen = ga_controller.evolve_population(env_template)

        # ga_controller.best_fitness is the best fitness seen across all generations so far by the controller
        current_overall_best_fitness = int(ga_controller.best_fitness)

        print(f"GA Gen {gen + 1}/{num_generations} | Mean Pop Fit: {mean_fitness:.2f} | Max This Gen: {max_fitness_this_gen:.2f} | Controller Best Overall: {current_overall_best_fitness}")

        if current_overall_best_fitness > highest_overall_fitness_on_disk:
            print(f"** New best overall GA fitness: {current_overall_best_fitness} (beats disk: {highest_overall_fitness_on_disk}). Saving model. **")

            # Remove old best model file if it exists
            if highest_overall_fitness_on_disk > -1:
                old_model_path = os.path.join(model_base_dir, f"{agent_prefix}_score_{highest_overall_fitness_on_disk}.pth")
                if os.path.exists(old_model_path):
                    try:
                        os.remove(old_model_path)
                        print(f"Removed old model: {old_model_path}")
                    except OSError as e:
                        print(f"Error removing old model {old_model_path}: {e}")

            # Save new best model
            new_model_filename = f"{agent_prefix}_score_{current_overall_best_fitness}.pth"
            new_model_path = os.path.join(model_base_dir, new_model_filename)
            ga_controller.save_best_individual(new_model_path)  # Pass the specific path
            highest_overall_fitness_on_disk = current_overall_best_fitness  # Update disk best

        if current_overall_best_fitness >= tetris_config.SCORE_TARGET:
            print(f"\nGA Target Score ({tetris_config.SCORE_TARGET}) reached!")
            break

    print("GA Training finished.")
    final_best_fitness = int(ga_controller.best_fitness)
    if final_best_fitness > highest_overall_fitness_on_disk:
        print(f"Final GA best fitness {final_best_fitness} is better than disk best {highest_overall_fitness_on_disk}. Saving.")
        if highest_overall_fitness_on_disk > -1:  # Remove old
            old_model_path = os.path.join(model_base_dir, f"{agent_prefix}_score_{highest_overall_fitness_on_disk}.pth")
            if os.path.exists(old_model_path): os.remove(old_model_path)
        final_model_path = os.path.join(model_base_dir, f"{agent_prefix}_score_{final_best_fitness}.pth")
        ga_controller.save_best_individual(final_model_path)
    print(f"Best GA fitness achieved this run: {final_best_fitness}")

def train_evolutionary_strategies(env_template: Tetris, cli_args_ns:argparse.Namespace, model_base_dir: str) -> None:
    print("\n--- Training Evolutionary Strategies (ES) ---")
    es_controller = ESAgent(
        state_size=tetris_config.STATE_SIZE, seed=tetris_config.SEED
    )

    num_generations = tetris_config.ES_N_GENERATIONS
    if cli_args_ns.num_generations is not None:
        num_generations = cli_args_ns.num_generations
        print(f"Overriding number of ES generations to: {num_generations}")

    agent_prefix = get_agent_file_prefix(cli_args_ns.agent_type)
    highest_overall_fitness_on_disk = find_best_existing_score(
        agent_prefix, model_base_dir
    )
    print(f"Initial best ES score on disk: {highest_overall_fitness_on_disk}")

    print(f"Starting ES Training for {num_generations} generations...")
    for gen in range(num_generations):
        mean_pop_fit, max_pop_fit_this_gen, central_fit = es_controller.evolve_step(
            env_template
        )

        # es_controller.current_best_fitness is the best fitness of the central parameters seen so far
        current_overall_best_fitness = int(es_controller.current_best_fitness)

        print(
            f"ES Gen {gen + 1}/{num_generations} | Mean Pop Fit: {mean_pop_fit:.2f} | Max This Gen: {max_pop_fit_this_gen:.2f} | Central Policy Fit: {central_fit:.2f} | Controller Best Overall: {current_overall_best_fitness}"
        )

        if current_overall_best_fitness > highest_overall_fitness_on_disk:
            print(
                f"** New best overall ES fitness: {current_overall_best_fitness} (beats disk: {highest_overall_fitness_on_disk}). Saving model. **"
            )

            if highest_overall_fitness_on_disk > -1:
                old_model_path = os.path.join(
                    model_base_dir,
                    f"{agent_prefix}_score_{highest_overall_fitness_on_disk}.pth",
                )
                if os.path.exists(old_model_path):
                    try:
                        os.remove(old_model_path)
                        print(f"Removed old model: {old_model_path}")
                    except OSError as e:
                        print(f"Error removing old model {old_model_path}: {e}")

            new_model_filename = (
                f"{agent_prefix}_score_{current_overall_best_fitness}.pth"
            )
            new_model_path = os.path.join(model_base_dir, new_model_filename)
            es_controller.save(new_model_path)  # Pass the specific path
            highest_overall_fitness_on_disk = current_overall_best_fitness

        # Print every N gens for logging progress, not tied to saving anymore
        if (gen + 1) % tetris_config.ES_PRINT_EVERY_GENS == 0:
            pass  # print statement above already covers detailed logging

        if current_overall_best_fitness >= tetris_config.ES_TARGET_GAME_SCORE:
            print(f"\nES Target Score ({tetris_config.ES_TARGET_GAME_SCORE}) reached!")
            break

    print("ES Training finished.")
    # Final check and save
    final_best_fitness = int(es_controller.current_best_fitness)
    if final_best_fitness > highest_overall_fitness_on_disk:
        print(
            f"Final ES best fitness {final_best_fitness} is better than disk best {highest_overall_fitness_on_disk}. Saving."
        )
        if highest_overall_fitness_on_disk > -1:  # Remove old
            old_model_path = os.path.join(
                model_base_dir,
                f"{agent_prefix}_score_{highest_overall_fitness_on_disk}.pth",
            )
            if os.path.exists(old_model_path):
                os.remove(old_model_path)

        final_model_path = os.path.join(
            model_base_dir, f"{agent_prefix}_score_{final_best_fitness}.pth"
        )
        es_controller.save(final_model_path)
    print(f"Best ES fitness achieved this run: {final_best_fitness}")


if __name__ == "__main__":
    opt = get_args()
    tetris_config.ensure_model_dir_exists()
    setup_seeds() # Set random seeds for reproducibility

    current_model_base_dir = tetris_config.MODEL_DIR
    print(f"Evolutionary models will be saved to: {current_model_base_dir}")

    env_template = Tetris(
        width=tetris_config.GAME_WIDTH,
        height=tetris_config.GAME_HEIGHT,
        block_size=tetris_config.GAME_BLOCK_SIZE,
    )

    if opt.agent_type == "genetic":
        train_genetic_algorithm(env_template, opt, current_model_base_dir)
    elif opt.agent_type == "es":
        train_evolutionary_strategies(env_template, opt, current_model_base_dir)
    else:
        print(f"Error: Agent type '{opt.agent_type}' is not supported by this evolutionary training script.")
