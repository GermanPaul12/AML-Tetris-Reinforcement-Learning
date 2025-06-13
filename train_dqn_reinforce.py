import os
import argparse
import config as tetris_config

from agents import DQNAgent, REINFORCEAgent

from helper import *
from src.tetris import Tetris

def get_args() -> argparse.Namespace:
    """Parses command-line arguments for training configuration."""
    parser = argparse.ArgumentParser("""Train DQN or REINFORCE Agents for Tetris""")
    parser.add_argument("--agent_type", type=str, default="dqn", choices=["dqn", "reinforce"], help="Type of agent to train (dqn or reinforce).",)
    parser.add_argument("--num_epochs", type=int, default=None, help="Total number of learning epochs (game-over learning steps for DQN, or completed episodes for REINFORCE).",)
    parser.add_argument("--render_game", action="store_true", help="Render the game during training.")
    args = parser.parse_args()
    return args

#######################################################
# Main Training Function for DQN and REINFORCE Agents #
#######################################################

def train(opt: argparse.Namespace):
    
    agent = opt.agent_type
    
    if agent == "dqn":
        print("\n--- Training DQN Agent ---")
        dqn_controller = DQNAgent(state_size=tetris_config.STATE_SIZE)
        
        total_learning_epochs = tetris_config.DQN_NUM_EPOCHS
        initial_epsilon = tetris_config.DQN_EPSILON_START
        final_epsilon = tetris_config.DQN_EPSILON_MIN
        num_decay_learning_steps = tetris_config.DQN_EPSILON_DECAY_EPOCHS
        print(f"DQN Epsilon will decay from {initial_epsilon} to {final_epsilon} over learning epochs.")
    elif agent == "reinforce":
        print("\n--- Training REINFORCE Agent ---")
        reinforce_controller = REINFORCEAgent(state_size=tetris_config.STATE_SIZE)
        
        total_learning_epochs = tetris_config.REINFORCE_TRAIN_GAMES
    else:
        raise ValueError(f"Unsupported agent type: {agent}. Choose 'dqn' or 'reinforce'.")

    if opt.num_epochs: total_learning_epochs = opt.num_epochs
    current_model_base_dir = tetris_config.MODEL_DIR
    print(f"Starting training. Target learning epochs/episodes: {total_learning_epochs}.")
    print(f"Models will be saved to '{current_model_base_dir}'.")

    # Set up training parameters
    current_epoch = 0
    games_played_this_run = 0
    current_game_score = 0
    total_score_all_games = 0.0
    highest_score_this_session = -1
    training_complete = False
    
    env = Tetris()
    state = env.reset()
    if tetris_config.DEVICE.type == "cuda": state = state.cuda()

    # Training Loop
    while current_epoch < total_learning_epochs and not training_complete:
        
        if agent == "dqn":
            epsilon = final_epsilon + (max(num_decay_learning_steps - current_epoch, 0) * (initial_epsilon - final_epsilon) / num_decay_learning_steps)
            action_tuple, state_info = dqn_controller.select_action(state, env, epsilon_override=epsilon)
        else:
            action_tuple, state_info = reinforce_controller.select_action(state, env)
        
        reward, game_over = env.step(action_tuple, render=opt.render_game)
        new_state = env.get_state_properties(env.board)
        if tetris_config.DEVICE.type == "cuda": new_state = new_state.cuda()
        current_game_score += reward
        
        if agent == "dqn": dqn_controller.expand_memory(reward=reward, done=game_over, state_info=state_info)
        else: reinforce_controller.expand_memory(reward=reward, state_info=state_info)

        if game_over:
            games_played_this_run += 1
            total_score_all_games += current_game_score
            avg_score = (total_score_all_games / games_played_this_run if games_played_this_run > 0 else 0.0)

            # Learning Step
            learned_this_epoch = False
            print_epsilon_str = ""
            loss_str = "Loss: N/A"
            
            if agent == "dqn":
                if len(dqn_controller.memory) >= tetris_config.DQN_BATCH_SIZE and len(dqn_controller.memory) >= (tetris_config.DQN_BUFFER_SIZE / 10):
                    experiences = dqn_controller.memory.sample()
                    if experiences[0].size(0) >= dqn_controller.memory.batch_size:
                        dqn_controller.learn_from_ReplayBuffer(experiences, tetris_config.DQN_GAMMA)
                        dqn_controller.learning_steps_done += 1
                        learned_this_epoch = True
                        loss_str = f"Loss: {float(dqn_controller.last_loss):.4f}"
                print_epsilon_str = f"| Epsilon: {epsilon:.4f}"
            else: # REINFORCE
                reinforce_controller.learn_episode()
                reinforce_controller.episodes_done += 1
                learned_this_epoch = True
                loss_str = f"Loss: {float(reinforce_controller.last_loss):.4f}"

            if learned_this_epoch: current_epoch += 1

            print(
                f"Epoch: {current_epoch}/{total_learning_epochs} | Game: {games_played_this_run} | "
                f"Score: {current_game_score} | Avg Score: {avg_score:.2f} | "
                f"Lines: {env.cleared_lines} {print_epsilon_str} | {loss_str}"
            )

            # Model Saving
            if current_game_score > highest_score_this_session:
                highest_score_this_session = current_game_score
                print(f"** New best score for this session: {highest_score_this_session}. Checking against disk. **")
                agent_prefix = get_agent_file_prefix(agent)
                disk_best_score = find_best_existing_score(agent_prefix, current_model_base_dir)
                if current_game_score > disk_best_score:
                    print(f"Current score {current_game_score} > disk best {agent} score {disk_best_score}. Saving.")
                    if disk_best_score > -1:
                        old_model_path = os.path.join(current_model_base_dir, f"{agent_prefix}_score_{disk_best_score}.pth")
                        if os.path.exists(old_model_path): os.remove(old_model_path)
                    new_model_path = os.path.join(current_model_base_dir, f"{agent_prefix}_score_{current_game_score}.pth")
                    if agent == "dqn": dqn_controller.save(new_model_path)
                    else: reinforce_controller.save(new_model_path)
                    print(f"Saved {agent} model (Score: {current_game_score}) to {new_model_path}")
                else:
                    print(f"Session best {current_game_score}, but disk best {agent} score is {disk_best_score}. Not overwriting disk.")

            if current_game_score >= tetris_config.EARLY_STOPPING_TARGET_SCORE:
                print(f"\nEarly stopping: Target score {tetris_config.EARLY_STOPPING_TARGET_SCORE} reached.")
                training_complete = True

            if agent == "dqn": dqn_controller.reset()
            else: reinforce_controller.reset()
            
            current_game_score = 0
            state = env.reset()
            if tetris_config.DEVICE.type == "cuda": state = state.cuda()
        else:
            state = new_state

    print("\nTraining finished.")
    if current_epoch >= total_learning_epochs: print(f"Completed {current_epoch} learning epochs/episodes as per configuration.")
    if highest_score_this_session > -1: print(f"Highest score achieved in this training session: {highest_score_this_session}")
    elif games_played_this_run > 0: print("No new high scores recorded in this session (or learning did not occur often).")
    else: print("No games or learning epochs were completed in this session.")

if __name__ == "__main__":
    opt = get_args()
    tetris_config.ensure_model_dir_exists()
    setup_seeds()
    
    train(opt)
