# tetris_rl_agents/main.py
import os
import sys
import subprocess
import config as tetris_config

def get_operation_choice():
    print("\nWhat would you like to do?")
    operations = ["Train agents", "Test a trained agent", "Evaluate agents"]
    for i, op_name in enumerate(operations): print(f"  {i + 1}. {op_name}")
    while True:
        try:
            choice = int(input(f"Choose an operation (1-{len(operations)}): ")) - 1
            if 0 <= choice < len(operations):
                return ["train", "test", "evaluate"][choice]
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_agent_selection(prompt, available_agents, allow_all=False, allow_multiple=False): # HIER MUSS NOCH MAL REFACTORED WERDEN
    print(f"\n{prompt}")
    display_agents = (
        [agent for agent in available_agents if agent != "random"]
        if "train" in prompt.lower()
        else list(available_agents)
    )

    options = list(display_agents)  # Make a mutable copy
    if allow_all and "all" not in options:  options.insert(0, "all")

    for i, agent_name in enumerate(options):
        display_name = agent_name.capitalize()
        if agent_name == "all":
            display_name = ("All trainable agents" if "train" in prompt.lower() else "All agents")
        print(f"  {i + 1}. {display_name}")

    chosen_agents_mapped = []
    while True:
        try:
            if allow_multiple:
                user_input = input(f"Choose agent(s) by number (e.g., 1 or 1,3 or type 'all' if available) (1-{len(options)}): ")
                if user_input == None: return  # Ensure input is not empty
                if "all" in user_input.lower().split(",") or '1' in user_input.split(","): user_input = "all"
                
                if allow_all and user_input.lower() == "all":
                    if "train" in prompt.lower():
                        chosen_agents_mapped = [agent for agent in tetris_config.AGENT_TYPES if agent != "random"]
                    else:
                        chosen_agents_mapped = ["all"]
                    break

                selected_indices = [int(x.strip()) - 1 for x in user_input.split(",")]
                current_selection_names = set()
                for idx in selected_indices:
                    if 0 <= (idx < len(options)):
                        current_selection_names.add(options[idx])
                    else:
                        break

                if current_selection_names:
                    chosen_agents_mapped = list(current_selection_names)
                    break
                else:
                    print("Invalid choice or empty selection. Please enter valid numbers from the list, separated by commas.")
            
            else:  # Single selection
                choice_idx = (int(input(f"Choose an agent by number (1-{len(options)}): ")) - 1)
                if 0 <= choice_idx < len(options):
                    agent_name_to_add = options[choice_idx]
                    if allow_all and agent_name_to_add == "all":
                        chosen_agents_mapped = ["all"]  # For scripts like evaluate.py
                    else:
                        chosen_agents_mapped = [agent_name_to_add]
                    break
                else:
                    print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter number(s).")
    return chosen_agents_mapped

def get_yes_no_input(prompt):
    while True:
        response = input(f"{prompt} (y/n): ").lower()
        if response in ["y", "yes"]: return True
        elif response in ["n", "no"]: return False
        else: print("Invalid input. Please enter 'y' or 'n'.")

def execute_script(command_list):
    print(f"\nExecuting command: {' '.join(command_list)}\n")
    try:
        process = subprocess.run(command_list, cwd=tetris_config.PROJECT_ROOT, check=False)
        if process.returncode != 0: print(f"\n--- Script exited with error (code: {process.returncode}) ---")
        else: print("\n--- Script finished successfully ---")
    except FileNotFoundError:
        print(f"Error: Could not find the script to execute. Command: {' '.join(command_list)}")
        print(f"Make sure the script exists at: {command_list[1]}")
    except Exception as e:
        print(f"An error occurred while trying to run the script: {e}")

def handle_training():
    print("\n--- Training Mode ---")

    trainable_agents = [agent for agent in tetris_config.AGENT_TYPES if agent != "random"]

    agents_to_train = get_agent_selection(
        "Which agent(s) would you like to train?",
        trainable_agents,
        allow_all=True,
        allow_multiple=True,
    )

    if not agents_to_train:
        print("No agents selected for training. Exiting training mode.")
        return

    if "all" in agents_to_train:  # "all" here refers to the string from options list
        agents_to_train = [agent for agent in tetris_config.AGENT_TYPES if agent != "random"]

    specific_num_epochs_dqn_reinforce = None
    specific_total_steps_onpolicy = None
    specific_num_gens_evolutionary = None
    
    if get_yes_no_input("Override default training duration (epochs/steps/generations) from config?"):
        print("Note: Duration parameters are specific to the training script being called.")
        specific_num_epochs_dqn_reinforce = input("Enter total learning epochs for DQN/REINFORCE (e.g., 3000) or Enter to skip: ").strip()
        specific_total_steps_onpolicy = input("Enter total environment steps for A2C/PPO (e.g., 1000000) or Enter to skip: ").strip()
        specific_num_gens_evolutionary = input("Enter number of generations for GA/ES (e.g., 200) or Enter to skip: ").strip()

    render_game = False
    if any(agent not in ["genetic", "es"] for agent in agents_to_train):
        if get_yes_no_input("Render game during training for step-based agents (DQN, REINFORCE, A2C, PPO)?"):
            render_game = True

    for agent_type in agents_to_train:
        print(f"\nPreparing to train: {agent_type.upper()}")
        command = [sys.executable]

        if agent_type in ["genetic", "es"]:
            command.append(os.path.join(tetris_config.PROJECT_ROOT, "train_evolutionary.py"))
            command.extend(["--agent_type", agent_type])
            if specific_num_gens_evolutionary: command.extend(["--num_generations", specific_num_gens_evolutionary])

        elif agent_type in ["dqn", "reinforce"]:
            command.append(os.path.join(tetris_config.PROJECT_ROOT, "train_dqn_reinforce.py"))
            command.extend(["--agent_type", agent_type])
            if specific_num_epochs_dqn_reinforce: command.extend(["--num_epochs", specific_num_epochs_dqn_reinforce])

        elif agent_type in ["a2c", "ppo"]:
            command.append(os.path.join(tetris_config.PROJECT_ROOT, "train_onpolicy.py"))
            command.extend(["--agent_type", agent_type])
            if specific_total_steps_onpolicy: command.extend(["--total_steps", specific_total_steps_onpolicy])
        
        if render_game: command.append("--render_game")

        else:
            print(f"Unknown or non-trainable agent type selected: {agent_type}. Skipping.")
            continue

        execute_script(command)

def handle_testing():
    print("\n--- Testing Mode ---")
    agent_to_test_list = get_agent_selection(
        "Which agent would you like to test?",
        tetris_config.AGENT_TYPES,
        allow_all=False,
        allow_multiple=False
    )
    if not agent_to_test_list:
        print("No agent selected. Exiting testing mode.")
        return
    agent_type = agent_to_test_list[0]
    if agent_type == "all":
        print("Testing 'all' agents simultaneously is not supported. Please select one.")
        return

    command = [sys.executable, os.path.join(tetris_config.PROJECT_ROOT, "test.py")]
    command.extend(["--agent_type", agent_type])

    num_games_str = input("Enter number of games to find the best run (default: 5): ").strip()
    if num_games_str:
        try:
            if int(num_games_str) > 0: command.extend(["--num_games", num_games_str])
        except ValueError:
            print("Invalid number for games, using default.")

    default_gif_basename = f"{agent_type}_best_game"
    output_gif_basename = input(f"Enter output GIF base name (default: {default_gif_basename}): ").strip()
    if not output_gif_basename: output_gif_basename = default_gif_basename
    command.extend(["--output_gif_basename", output_gif_basename])

    execute_script(command)


def handle_evaluation():
    print("\n--- Evaluation Mode ---")
    agents_to_evaluate = get_agent_selection(
        "Which agent(s) to evaluate?",
        tetris_config.AGENT_TYPES,
        allow_all=True,
        allow_multiple=True
    )

    if not agents_to_evaluate:
        print("No agents selected. Exiting evaluation mode.")
        return

    command = [sys.executable, os.path.join(tetris_config.PROJECT_ROOT, "evaluate.py")]    
    command.extend(["--agent_types", ",".join(agents_to_evaluate)])

    num_eval_games_str = input(f"Enter number of evaluation games per agent (default: {tetris_config.NUM_EVAL_GAMES}): ").strip()
    if num_eval_games_str:
        try:
            if int(num_eval_games_str) > 0: command.extend(["--num_eval_games", num_eval_games_str])
        except ValueError:
            print("Invalid number for eval games, using default.")

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
    print("Welcome to the Tetris RL Agents Manager!")
    tetris_config.ensure_model_dir_exists()
    main_interactive_loop()
