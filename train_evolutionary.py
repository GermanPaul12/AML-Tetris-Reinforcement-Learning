# tetris_rl_agents/train_evolutionary.py
import argparse
import os
import random
import re # Added for regex in parsing filenames
import numpy as np
import torch

import config as tetris_config
#import test_config as tetris_config # For testing with a different config
from agents import GeneticAlgorithmController, ESAgent # Import controllers
from src.tetris import Tetris # The game environment

# --- Helper Functions for Model Saving (Copied from train.py for now, ideally in a shared util) ---
def get_agent_file_prefix(agent_type_str, is_actor=False, is_critic=False):
    """Gets the filename prefix for the agent's model file."""
    processed_agent_type = agent_type_str.replace("_", "-") 
    if agent_type_str == "ppo": 
        if is_actor: return "ppo-actor"
        elif is_critic: return "ppo-critic"
        else: return "ppo-model" 
    return processed_agent_type

def parse_score_from_filename(filename_basename, expected_prefix):
    """
    Parses the score from a model filename.
    Expected format: {expected_prefix}_score_{score}.pth
    """
    pattern = re.compile(f"^{re.escape(expected_prefix)}_score_(\\d+)\\.pth$")
    match = pattern.match(filename_basename)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None

def find_best_existing_score(agent_prefix, model_dir):
    """
    Finds the highest score from existing model files for a given agent prefix in the model_dir.
    Returns the score (integer), or -1 if no relevant files are found.
    """
    max_score = -1 
    if not os.path.isdir(model_dir):
        try:
            os.makedirs(model_dir, exist_ok=True)
        except OSError:
            print(f"Warning: Model directory {model_dir} does not exist and could not be created.")
            return max_score 
            
    for filename in os.listdir(model_dir):
        score = parse_score_from_filename(filename, agent_prefix)
        if score is not None and score > max_score:
            max_score = score
    return max_score
# --- End Helper Functions ---


def get_args():
    parser = argparse.ArgumentParser(
        """Train Evolutionary Agents (GA, ES) for Tetris""")
    
    parser.add_argument("--agent_type", type=str, required=True,
                        choices=["genetic", "es"],
                        help="Type of evolutionary agent to train.")
    parser.add_argument("--fps", type=int, default=300, 
                        help="Frames per second for video (if generated, not used in training).")
    parser.add_argument("--num_generations", type=int, default=None,
                        help="Number of generations to train (overrides agent's config).")
    
    args = parser.parse_args()
    return args

def setup_seeds():
    torch.manual_seed(tetris_config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tetris_config.SEED)
    np.random.seed(tetris_config.SEED)
    random.seed(tetris_config.SEED)
    print(f"Seeds set to: {tetris_config.SEED}")

def train_genetic_algorithm(env_template: Tetris, cli_args_ns, model_base_dir: str):
    print("\n--- Training Genetic Algorithm (GA) ---")
    ga_controller = GeneticAlgorithmController(
        state_size=tetris_config.STATE_SIZE,
        seed=tetris_config.SEED
    )
    
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
            ga_controller.save_best_individual(new_model_path) # Pass the specific path
            highest_overall_fitness_on_disk = current_overall_best_fitness # Update disk best
        
        if current_overall_best_fitness >= tetris_config.SCORE_TARGET:
            print(f"\nGA Target Score ({tetris_config.SCORE_TARGET}) reached!")
            break
            
    print("GA Training finished.")
    # Final check and save if controller's best is better than what's on disk (e.g. if loop finished by num_generations)
    final_best_fitness = int(ga_controller.best_fitness)
    if final_best_fitness > highest_overall_fitness_on_disk:
        print(f"Final GA best fitness {final_best_fitness} is better than disk best {highest_overall_fitness_on_disk}. Saving.")
        if highest_overall_fitness_on_disk > -1: # Remove old
            old_model_path = os.path.join(model_base_dir, f"{agent_prefix}_score_{highest_overall_fitness_on_disk}.pth")
            if os.path.exists(old_model_path): os.remove(old_model_path)
        
        final_model_path = os.path.join(model_base_dir, f"{agent_prefix}_score_{final_best_fitness}.pth")
        ga_controller.save_best_individual(final_model_path)
    print(f"Best GA fitness achieved this run: {final_best_fitness}")


def train_evolutionary_strategies(env_template: Tetris, cli_args_ns, model_base_dir: str):
    print("\n--- Training Evolutionary Strategies (ES) ---")
    es_controller = ESAgent(
        state_size=tetris_config.STATE_SIZE,
        seed=tetris_config.SEED
    )

    num_generations = tetris_config.ES_N_GENERATIONS
    if cli_args_ns.num_generations is not None:
        num_generations = cli_args_ns.num_generations
        print(f"Overriding number of ES generations to: {num_generations}")

    agent_prefix = get_agent_file_prefix(cli_args_ns.agent_type)
    highest_overall_fitness_on_disk = find_best_existing_score(agent_prefix, model_base_dir)
    print(f"Initial best ES score on disk: {highest_overall_fitness_on_disk}")

    print(f"Starting ES Training for {num_generations} generations...")
    for gen in range(num_generations):
        mean_pop_fit, max_pop_fit_this_gen, central_fit = es_controller.evolve_step(env_template)
        
        # es_controller.current_best_fitness is the best fitness of the central parameters seen so far
        current_overall_best_fitness = int(es_controller.current_best_fitness)

        print(f"ES Gen {gen + 1}/{num_generations} | Mean Pop Fit: {mean_pop_fit:.2f} | Max This Gen: {max_pop_fit_this_gen:.2f} | Central Policy Fit: {central_fit:.2f} | Controller Best Overall: {current_overall_best_fitness}")

        if current_overall_best_fitness > highest_overall_fitness_on_disk:
            print(f"** New best overall ES fitness: {current_overall_best_fitness} (beats disk: {highest_overall_fitness_on_disk}). Saving model. **")
            
            if highest_overall_fitness_on_disk > -1:
                old_model_path = os.path.join(model_base_dir, f"{agent_prefix}_score_{highest_overall_fitness_on_disk}.pth")
                if os.path.exists(old_model_path):
                    try:
                        os.remove(old_model_path)
                        print(f"Removed old model: {old_model_path}")
                    except OSError as e:
                        print(f"Error removing old model {old_model_path}: {e}")
            
            new_model_filename = f"{agent_prefix}_score_{current_overall_best_fitness}.pth"
            new_model_path = os.path.join(model_base_dir, new_model_filename)
            es_controller.save(new_model_path) # Pass the specific path
            highest_overall_fitness_on_disk = current_overall_best_fitness
        
        # Print every N gens for logging progress, not tied to saving anymore
        if (gen + 1) % tetris_config.ES_PRINT_EVERY_GENS == 0:
            pass # print statement above already covers detailed logging

        if current_overall_best_fitness >= tetris_config.ES_TARGET_GAME_SCORE:
            print(f"\nES Target Score ({tetris_config.ES_TARGET_GAME_SCORE}) reached!")
            break
    
    print("ES Training finished.")
    # Final check and save
    final_best_fitness = int(es_controller.current_best_fitness)
    if final_best_fitness > highest_overall_fitness_on_disk:
        print(f"Final ES best fitness {final_best_fitness} is better than disk best {highest_overall_fitness_on_disk}. Saving.")
        if highest_overall_fitness_on_disk > -1: # Remove old
            old_model_path = os.path.join(model_base_dir, f"{agent_prefix}_score_{highest_overall_fitness_on_disk}.pth")
            if os.path.exists(old_model_path): os.remove(old_model_path)
        
        final_model_path = os.path.join(model_base_dir, f"{agent_prefix}_score_{final_best_fitness}.pth")
        es_controller.save(final_model_path)
    print(f"Best ES fitness achieved this run: {final_best_fitness}")


if __name__ == "__main__":
    cli_args = get_args()
    tetris_config.ensure_model_dir_exists() # Ensures MODEL_DIR exists
    setup_seeds()

    # Determine base directory for saving models (handles test_suite mode)
    current_model_base_dir = tetris_config.MODEL_DIR
    if 'test_config' in str(tetris_config) and hasattr(tetris_config, 'PROJECT_ROOT'): # A bit fragile check
        # This assumes your test_config might set PROJECT_ROOT differently or you want a subfolder.
        # For consistency with train.py, ensure test_config also defines MODEL_DIR if it's different.
        # A simpler way: if a global TEST_MODE flag is set, then use a different dir.
        # For now, let's assume test_config.MODEL_DIR would be used if test_config is imported.
        # The provided setup implies tetris_config is what's imported, so MODEL_DIR comes from there.
        # If you use "import test_config as tetris_config", then tetris_config.MODEL_DIR will point to test's one.
        # The most robust way for test suites is to have a separate models_test_suite directory.
        
        # Let's refine the test suite directory logic for clarity
        if hasattr(tetris_config, 'PROJECT_ROOT') and "models_test_suite" in tetris_config.MODEL_DIR: # if MODEL_DIR itself is the test dir
            current_model_base_dir = tetris_config.MODEL_DIR
        elif hasattr(tetris_config, 'PROJECT_ROOT'): # Fallback to a standard test suite subfolder if not specified in MODEL_DIR
             #This part might be redundant if your test_config.py directly sets MODEL_DIR to the test model path
             #If 'test_config' is imported as tetris_config, then tetris_config.MODEL_DIR already IS the test model dir
            pass # current_model_base_dir will be from the active config

    print(f"Evolutionary models will be saved to: {current_model_base_dir}")


    env_template = Tetris(width=tetris_config.GAME_WIDTH,
                          height=tetris_config.GAME_HEIGHT,
                          block_size=tetris_config.GAME_BLOCK_SIZE)

    if cli_args.agent_type == "genetic":
        train_genetic_algorithm(env_template, cli_args, current_model_base_dir)
    elif cli_args.agent_type == "es":
        train_evolutionary_strategies(env_template, cli_args, current_model_base_dir)
    else:
        print(f"Error: Agent type '{cli_args.agent_type}' is not supported by this evolutionary training script.")