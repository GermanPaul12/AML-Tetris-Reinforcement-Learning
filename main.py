# tetris_rl_agents/main.py
import subprocess
import sys
import os
import config as tetris_config


def get_operation_choice():
    print("\nWhat would you like to do?")
    operations = ["Train agents", "Test a trained agent", "Evaluate agents"]
    for i, op_name in enumerate(operations):
        print(f"  {i + 1}. {op_name}")
    while True:
        try:
            choice = int(input(f"Choose an operation (1-{len(operations)}): ")) - 1
            if 0 <= choice < len(operations):
                return ["train", "test", "evaluate"][choice]
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_agent_selection(
    prompt, available_agents, allow_all=False, allow_multiple=False
):
    print(f"\n{prompt}")
    # Use a filtered list of agents if needed, e.g., exclude 'random' for training
    display_agents = (
        [agent for agent in available_agents if agent != "random"]
        if "train" in prompt.lower()
        else list(available_agents)
    )

    options = list(display_agents)  # Make a mutable copy
    if allow_all and "all" not in options:  # Ensure "all" is only added once
        options.insert(0, "all")

    for i, agent_name in enumerate(options):
        # Capitalize for display, handle "all" specially if it was inserted
        display_name = agent_name.capitalize()
        if agent_name == "all":
            display_name = (
                "All trainable agents" if "train" in prompt.lower() else "All agents"
            )
        print(f"  {i + 1}. {display_name}")

    chosen_agents_mapped = []
    while True:
        try:
            if allow_multiple:
                user_input = input(
                    f"Choose agent(s) by number (e.g., 1 or 1,3 or type 'all' if available) (1-{len(options)}): "
                )
                if allow_all and user_input.lower() == "all":
                    # Map "all" to the actual list of agents, excluding "random" for training
                    if "train" in prompt.lower():
                        chosen_agents_mapped = [
                            agent
                            for agent in tetris_config.AGENT_TYPES
                            if agent != "random"
                        ]
                    else:
                        chosen_agents_mapped = [
                            "all"
                        ]  # Scripts like evaluate.py might handle "all"
                    break

                selected_indices = [int(x.strip()) - 1 for x in user_input.split(",")]

                current_selection_names = []
                valid_selection = True
                for idx in selected_indices:
                    if 0 <= idx < len(options):
                        agent_name_to_add = options[idx]
                        if (
                            allow_all and agent_name_to_add == "all"
                        ):  # User selected the number for "all"
                            if "train" in prompt.lower():
                                current_selection_names.extend(
                                    [
                                        agent
                                        for agent in tetris_config.AGENT_TYPES
                                        if agent != "random"
                                    ]
                                )
                            else:
                                current_selection_names.append(
                                    "all"
                                )  # evaluate.py might handle this
                            # If "all" is selected with others, "all" takes precedence for training loop
                            # Or, decide how to handle combined "all" + specific. Here, if "all" is chosen, it's "all".
                            # To make it truly "all" if the number for all is selected:
                            # chosen_agents_mapped = [agent for agent in tetris_config.AGENT_TYPES if agent != "random"] if "train" in prompt.lower() else ["all"]
                            # break # Exit while loop after processing "all"
                            continue  # Continue to collect other selections if "all" is just one of many
                        current_selection_names.append(agent_name_to_add)
                    else:
                        valid_selection = False
                        break

                if valid_selection and current_selection_names:
                    seen = set()
                    chosen_agents_mapped = [
                        x
                        for x in current_selection_names
                        if not (x in seen or seen.add(x))
                    ]
                    if (
                        "all" in chosen_agents_mapped and "train" in prompt.lower()
                    ):  # if 'all' (the string from options) got selected
                        chosen_agents_mapped = [
                            agent
                            for agent in tetris_config.AGENT_TYPES
                            if agent != "random"
                        ]

                    break
                else:
                    print(
                        "Invalid choice or empty selection. Please enter valid numbers from the list, separated by commas."
                    )

            else:  # Single selection
                choice_idx = (
                    int(input(f"Choose an agent by number (1-{len(options)}): ")) - 1
                )
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
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def execute_script(command_list):
    print(f"\nExecuting command: {' '.join(command_list)}\n")
    try:
        # Using Popen for potentially long-running processes if we wanted to do more,
        # but run is fine for sequential execution.
        process = subprocess.run(
            command_list, cwd=tetris_config.PROJECT_ROOT, check=False
        )
        if process.returncode != 0:
            print(f"\n--- Script exited with error (code: {process.returncode}) ---")
        else:
            print("\n--- Script finished successfully ---")
    except FileNotFoundError:
        print(
            f"Error: Could not find the script to execute. Command: {' '.join(command_list)}"
        )
        print(f"Make sure the script exists at: {command_list[1]}")
    except Exception as e:
        print(f"An error occurred while trying to run the script: {e}")


def handle_training():
    print("\n--- Training Mode ---")
    # For training, "random" agent is not trainable.
    trainable_agents = [
        agent for agent in tetris_config.AGENT_TYPES if agent != "random"
    ]

    agents_to_train = get_agent_selection(
        "Which agent(s) would you like to train?",
        trainable_agents,  # Pass only trainable agents
        allow_all=True,
        allow_multiple=True,
    )

    if not agents_to_train:
        print("No agents selected for training. Exiting training mode.")
        return

    # Consolidate "all" if it was selected numerically among others
    if "all" in agents_to_train:  # "all" here refers to the string from options list
        agents_to_train = [
            agent for agent in tetris_config.AGENT_TYPES if agent != "random"
        ]

    override_duration = get_yes_no_input(
        "Override default training duration (epochs/steps/generations) from config?"
    )

    specific_num_epochs_dqn_reinforce = (
        None  # For train_dqn_reinforce.py's --num_epochs
    )
    specific_total_steps_onpolicy = None  # For train_onpolicy.py's --total_steps
    specific_num_gens_evolutionary = (
        None  # For train_evolutionary.py's --num_generations
    )

    if override_duration:
        print(
            "Note: Duration parameters are specific to the training script being called."
        )
        specific_num_epochs_dqn_reinforce = input(
            "Enter total learning epochs for DQN/REINFORCE (e.g., 3000) or Enter to skip: "
        ).strip()
        specific_total_steps_onpolicy = input(
            "Enter total environment steps for A2C/PPO (e.g., 1000000) or Enter to skip: "
        ).strip()
        specific_num_gens_evolutionary = input(
            "Enter number of generations for GA/ES (e.g., 200) or Enter to skip: "
        ).strip()

    render_game = False
    # Check if any selected agent is not GA/ES to offer rendering
    if any(agent not in ["genetic", "es"] for agent in agents_to_train):
        if get_yes_no_input(
            "Render game during training for step-based agents (DQN, REINFORCE, A2C, PPO)?"
        ):
            render_game = True

    for agent_type in agents_to_train:
        print(f"\nPreparing to train: {agent_type.upper()}")
        command = [sys.executable]

        if agent_type in ["genetic", "es"]:
            command.append(
                os.path.join(tetris_config.PROJECT_ROOT, "train_evolutionary.py")
            )
            command.extend(["--agent_type", agent_type])
            if override_duration and specific_num_gens_evolutionary:
                command.extend(["--num_generations", specific_num_gens_evolutionary])

        elif agent_type in ["dqn", "reinforce"]:
            command.append(
                os.path.join(tetris_config.PROJECT_ROOT, "train_dqn_reinforce.py")
            )
            command.extend(["--agent_type", agent_type])
            if override_duration and specific_num_epochs_dqn_reinforce:
                command.extend(["--num_epochs", specific_num_epochs_dqn_reinforce])
            if render_game:
                command.append("--render_game")

        elif agent_type in ["a2c", "ppo"]:
            command.append(
                os.path.join(tetris_config.PROJECT_ROOT, "train_onpolicy.py")
            )  # Or train_a2c_ppo.py
            command.extend(["--agent_type", agent_type])
            if override_duration and specific_total_steps_onpolicy:
                command.extend(["--total_steps", specific_total_steps_onpolicy])
            # Can also add --num_games as an option for train_onpolicy.py
            if render_game:
                command.append("--render_game")

        else:
            print(
                f"Unknown or non-trainable agent type selected: {agent_type}. Skipping."
            )
            continue

        execute_script(command)


def handle_testing():
    print("\n--- Testing Mode ---")
    agent_to_test_list = get_agent_selection(
        "Which agent would you like to test?",
        tetris_config.AGENT_TYPES,  # All agents can be tested
        allow_all=False,
        allow_multiple=False,
    )
    if not agent_to_test_list:
        print("No agent selected. Exiting testing mode.")
        return
    agent_type = agent_to_test_list[0]
    if agent_type == "all":  # Should not happen if allow_all=False
        print(
            "Testing 'all' agents simultaneously is not supported. Please select one."
        )
        return

    command = [sys.executable, os.path.join(tetris_config.PROJECT_ROOT, "test.py")]
    command.extend(["--agent_type", agent_type])

    # Default value for num_games now comes from test.py's argparse (5)
    # The config value can still be used for the prompt's text.
    num_games_str = input(
        "Enter number of games to find the best run (default: 5): "
    ).strip()
    if num_games_str:
        try:
            if int(num_games_str) > 0:
                command.extend(["--num_games", num_games_str])
        except ValueError:
            print("Invalid number for games, using default.")

    # FPS argument is removed as it's no longer used by test.py

    default_gif_basename = f"{agent_type}_best_game"
    output_gif_basename = input(
        f"Enter output GIF base name (default: {default_gif_basename}): "
    ).strip()
    if not output_gif_basename:
        output_gif_basename = default_gif_basename
    command.extend(["--output_gif_basename", output_gif_basename])

    execute_script(command)


def handle_evaluation():
    print("\n--- Evaluation Mode ---")
    agents_to_evaluate = get_agent_selection(
        "Which agent(s) to evaluate?",
        tetris_config.AGENT_TYPES,  # All agents can be evaluated
        allow_all=True,
        allow_multiple=True,
    )

    if not agents_to_evaluate:
        print("No agents selected. Exiting evaluation mode.")
        return

    command = [sys.executable, os.path.join(tetris_config.PROJECT_ROOT, "evaluate.py")]

    if "all" in agents_to_evaluate:
        command.extend(["--agent_types", "all"])
    else:
        command.extend(["--agent_types", ",".join(agents_to_evaluate)])

    num_eval_games_str = input(
        f"Enter number of evaluation games per agent (default: {tetris_config.NUM_EVAL_GAMES}): "
    ).strip()
    if num_eval_games_str:
        try:
            if int(num_eval_games_str) > 0:
                command.extend(["--num_eval_games", num_eval_games_str])
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
