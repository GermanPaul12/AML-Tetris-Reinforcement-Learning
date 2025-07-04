import os
import argparse
import config

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

def train_evolutionary_agents(env_template: Tetris, opt: argparse.Namespace, model_base_dir: str) -> None:
    """Train evolutionary agents based on the specified type.
    
    Args:
        env_template (Tetris): The Tetris environment template to use for training.
        opt (argparse.Namespace): Parsed command line arguments.
        model_base_dir (str): Directory where models will be saved.
    """
    
    agent = opt.agent_type
    
    if agent == "genetic":
        print("\n--- Training Genetic Algorithm (GA) ---")
        ga_controller = GeneticAlgorithmController(state_size=config.STATE_SIZE)

        num_generations = config.GA_N_GENERATIONS
        if opt.num_generations:
            num_generations = opt.num_generations
            print(f"Overriding number of GA generations to: {num_generations}")    
    elif agent == "es":
        print("\n--- Training Evolutionary Strategies (ES) ---")
        es_controller = ESAgent(state_size=config.STATE_SIZE)

        num_generations = config.ES_N_GENERATIONS
        if opt.num_generations:
            num_generations = opt.num_generations
            print(f"Overriding number of ES generations to: {num_generations}")
    else:
        raise ValueError(f"Unsupported agent type: {agent}")
    
    agent_prefix = get_agent_file_prefix(agent)
    highest_overall_fitness_on_disk = find_best_existing_score(agent_prefix, model_base_dir)
    print(f"Initial best Agent score on disk: {highest_overall_fitness_on_disk}")

    print(f"Starting Training for {num_generations} generations...")
    for gen in range(num_generations):
        
        if agent == "genetic":
            mean_pop_fit, max_pop_fit_this_gen = ga_controller.evolve_population(env_template)
            current_overall_best_fitness = int(ga_controller.best_fitness)
            print(f"GA Gen {gen + 1}/{num_generations} | Mean Pop Fit: {mean_pop_fit:.2f} | Max This Gen: {max_pop_fit_this_gen:.2f} | Controller Best Overall: {current_overall_best_fitness}")
        elif agent == "es":
            mean_pop_fit, max_pop_fit_this_gen, central_fit = es_controller.learn(env_template)
            current_overall_best_fitness = int(es_controller.current_best_fitness)
            print(f"ES Gen {gen + 1}/{num_generations} | Mean Pop Fit: {mean_pop_fit:.2f} | Max This Gen: {max_pop_fit_this_gen:.2f} | Central Policy Fit: {central_fit:.2f} | Controller Best Overall: {current_overall_best_fitness}")
        
        if current_overall_best_fitness > highest_overall_fitness_on_disk:
            print(f"** New best overall fitness: {current_overall_best_fitness} (beats disk: {highest_overall_fitness_on_disk}). Saving model. **")
            
            # Remove old best model if it exists
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
            if agent == "genetic": ga_controller.save_best_individual(new_model_path)
            elif agent == "es": es_controller.save(new_model_path)
            highest_overall_fitness_on_disk = current_overall_best_fitness

        # Save if tetris has been beaten
        if current_overall_best_fitness >= config.SCORE_TARGET:
            print(f"\nTarget Score ({config.SCORE_TARGET}) reached!")
            break

    print(f"Best ES fitness achieved this run: {current_overall_best_fitness}")

if __name__ == "__main__":
    opt = get_args()
    config.ensure_model_dir_exists()
    setup_seeds()

    current_model_base_dir = config.MODEL_DIR
    print(f"Evolutionary models will be saved to: {current_model_base_dir}")

    env_template = Tetris()

    train_evolutionary_agents(env_template, opt, current_model_base_dir)
