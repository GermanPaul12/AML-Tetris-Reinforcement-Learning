# tetris_rl_agents/main.py
import subprocess
import sys
import os

# Assuming config.py is in the same directory or Python path is set up
import config as tetris_config
#import test_config as tetris_config

def get_operation_choice():
    print("\nWhat would you like to do?")
    operations = ["Train agents", "Test a trained agent", "Evaluate agents"]
    for i, op_name in enumerate(operations):
        print(f"  {i+1}. {op_name}")
    while True:
        try:
            choice = int(input(f"Choose an operation (1-{len(operations)}): ")) - 1
            if 0 <= choice < len(operations):
                return ["train", "test", "evaluate"][choice]
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_agent_selection(prompt, available_agents, allow_all=False, allow_multiple=False):
    print(f"\n{prompt}")
    options = list(available_agents) # Make a mutable copy
    if allow_all:
        options.insert(0, "all (recommended for evaluation)")
    
    for i, agent_name in enumerate(options):
        print(f"  {i+1}. {agent_name.capitalize()}")

    chosen_agents = []
    while True:
        try:
            if allow_multiple:
                user_input = input(f"Choose agent(s) by number (e.g., 1 or 1,3 or type 'all' if available) (1-{len(options)}): ")
                if allow_all and user_input.lower() == "all":
                    return ["all"] # Special keyword for scripts that handle "all"

                selected_indices = [int(x.strip()) - 1 for x in user_input.split(',')]
                
                valid_selection = True
                current_selection_names = []
                for idx in selected_indices:
                    if 0 <= idx < len(options):
                        # Map back to original agent name if "all" was an option
                        agent_name_to_add = options[idx]
                        if allow_all and agent_name_to_add.startswith("all"): # "all" option itself
                            return ["all"] # If user typed the number for "all"
                        current_selection_names.append(agent_name_to_add)
                    else:
                        valid_selection = False
                        break
                
                if valid_selection and current_selection_names:
                    # Remove duplicates while preserving order (if that matters, otherwise set is fine)
                    seen = set()
                    chosen_agents = [x for x in current_selection_names if not (x in seen or seen.add(x))]
                    break
                else:
                    print("Invalid choice or empty selection. Please enter valid numbers from the list, separated by commas.")

            else: # Single selection
                choice_idx = int(input(f"Choose an agent by number (1-{len(options)}): ")) - 1
                if 0 <= choice_idx < len(options):
                    agent_name_to_add = options[choice_idx]
                    if allow_all and agent_name_to_add.startswith("all"):
                         chosen_agents = ["all"]
                    else:
                        chosen_agents = [agent_name_to_add]
                    break
                else:
                    print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter number(s).")
    return chosen_agents


def get_yes_no_input(prompt):
    while True:
        response = input(f"{prompt} (y/n): ").lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

def execute_script(command_list):
    print(f"\nExecuting command: {' '.join(command_list)}\n")
    try:
        process = subprocess.run(command_list, cwd=tetris_config.PROJECT_ROOT, check=False)
        if process.returncode != 0:
            print(f"\n--- Script exited with error (code: {process.returncode}) ---")
        else:
            print(f"\n--- Script finished successfully ---")
    except FileNotFoundError:
        print(f"Error: Could not find the script to execute. Command: {' '.join(command_list)}")
    except Exception as e:
        print(f"An error occurred while trying to run the script: {e}")


def handle_training():
    print("\n--- Training Mode ---")
    agents_to_train = get_agent_selection(
        "Which agent(s) would you like to train?",
        tetris_config.AGENT_TYPES,
        allow_all=True, # Allow selecting "all" which we'll interpret as loop
        allow_multiple=True
    )

    if not agents_to_train:
        print("No agents selected for training. Exiting training mode.")
        return

    specific_num_pieces = None
    specific_num_gens = None
    render_game = False

    if get_yes_no_input("Override default training duration (pieces/generations) from config?"):
        specific_num_pieces = input("Enter number of total pieces (for step-based agents, e.g., DQN, PPO) or press Enter to skip: ")
        specific_num_gens = input("Enter number of generations (for GA/ES) or press Enter to skip: ")

    if any(agent not in ["genetic", "es", "random"] for agent in agents_to_train) or \
       ("all" in agents_to_train and any(a not in ["genetic", "es", "random"] for a in tetris_config.AGENT_TYPES)):
        if get_yes_no_input("Render game during training for step-based agents? (not recommended for GA/ES eval phases)"):
            render_game = True
    
    agents_for_loop = []
    if "all" in agents_to_train:
        agents_for_loop = [agent for agent in tetris_config.AGENT_TYPES if agent != "random"]
    else:
        agents_for_loop = [agent for agent in agents_to_train if agent != "random"]


    for agent_type in agents_for_loop:
        print(f"\nPreparing to train: {agent_type.upper()}")
        command = [sys.executable]
        
        if agent_type in ["genetic", "es"]:
            command.append(os.path.join(tetris_config.PROJECT_ROOT, "train_evolutionary.py"))
            command.extend(["--agent_type", agent_type])
            if specific_num_gens and specific_num_gens.strip():
                command.extend(["--num_generations", specific_num_gens])
        elif agent_type in tetris_config.AGENT_TYPES: # Step-based RL agents
            command.append(os.path.join(tetris_config.PROJECT_ROOT, "train.py"))
            command.extend(["--agent_type", agent_type])
            if specific_num_pieces and specific_num_pieces.strip():
                command.extend(["--num_total_pieces", specific_num_pieces])
            if render_game:
                command.append("--render_game") # Enable rendering for step-based agents
        else:
            print(f"Unknown agent type for training: {agent_type}. Skipping.")
            continue
        
        execute_script(command)

def handle_testing():
    print("\n--- Testing Mode ---")
    # For testing, usually one agent at a time
    agent_to_test_list = get_agent_selection(
        "Which agent would you like to test?",
        tetris_config.AGENT_TYPES,
        allow_all=False,
        allow_multiple=False
    )
    if not agent_to_test_list:
        print("No agent selected for testing. Exiting testing mode.")
        return
    agent_type = agent_to_test_list[0]

    command = [sys.executable, os.path.join(tetris_config.PROJECT_ROOT, "test.py")]
    command.extend(["--agent_type", agent_type])
    
    num_games_str = input(f"Enter number of games for video (default: {tetris_config.NUM_TEST_RUNS_GIF}): ").strip()
    if num_games_str:
        command.extend(["--num_games", num_games_str])
        
    fps_str = input(f"Enter FPS for video (default: {tetris_config.GIF_FPS}): ").strip()
    if fps_str:
        command.extend(["--fps", fps_str])
        
    default_video_name = f"{agent_type}_test_video.mp4"
    output_video = input(f"Enter output video filename (default: {default_video_name}): ").strip()
    if not output_video:
        output_video = default_video_name
    command.extend(["--output_video", output_video])
    
    execute_script(command)

def handle_evaluation():
    print("\n--- Evaluation Mode ---")
    agents_to_evaluate = get_agent_selection(
        "Which agent(s) to evaluate?",
        tetris_config.AGENT_TYPES,
        allow_all=True,
        allow_multiple=True # evaluate.py can take comma-separated list or "all"
    )

    if not agents_to_evaluate:
        print("No agents selected for evaluation. Exiting evaluation mode.")
        return

    command = [sys.executable, os.path.join(tetris_config.PROJECT_ROOT, "evaluate.py")]
    
    # evaluate.py expects --agent_types as a comma-separated string or "all"
    if "all" in agents_to_evaluate:
        command.extend(["--agent_types", "all"])
    else:
        command.extend(["--agent_types", ",".join(agents_to_evaluate)])

    num_eval_games_str = input(f"Enter number of evaluation games per agent (default: {tetris_config.NUM_EVAL_GAMES}): ").strip()
    if num_eval_games_str:
        command.extend(["--num_eval_games", num_eval_games_str])
        
    execute_script(command)

def main_interactive_loop():
    while True:
        operation = get_operation_choice()

        if operation == "train":
            handle_training()
        elif operation == "test":
            handle_testing()
        elif operation == "evaluate":
            handle_evaluation()
        
        if not get_yes_no_input("\nPerform another operation?"):
            print("Exiting application.")
            break

if __name__ == "__main__":
    # This script is now primarily interactive.
    # If you still want to support direct CLI args for operation like before:
    # You could parse a single "operation" arg here and then call the respective handler.
    # But for fully interactive, the loop above is fine.

    print("Welcome to the Tetris RL Agents Manager!")
    tetris_config.ensure_model_dir_exists() # Ensure model directory exists at start
    main_interactive_loop()