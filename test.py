# tetris_rl_agents/test.py
import argparse
import torch
import cv2 # For video writing
import os
import random
import numpy as np

# New imports for multi-agent structure
import config as tetris_config # Import the project-specific config
from agents import AGENT_REGISTRY # To get agent classes
from src.tetris import Tetris # The game environment

def get_args():
    parser = argparse.ArgumentParser(
        """Test pre-trained Reinforcement Learning Agents for Tetris""")

    parser.add_argument("--agent_type", type=str, default="dqn",
                        choices=list(tetris_config.AGENT_TYPES),
                        help="Type of agent to test.")
    parser.add_argument("--output_video", type=str, default="output_tetris.mp4",
                        help="Name of the output video file.")
    parser.add_argument("--fps", type=int, default=30, 
                        help="Frames per second for the video. Higher for faster playback.")
    parser.add_argument("--num_games", type=int, default=tetris_config.NUM_TEST_RUNS_GIF,
                        help="Number of games to play and record in the video.")

    args = parser.parse_args()
    return args

def test():
    opt = get_args()
    tetris_config.ensure_model_dir_exists() # Though not saving, good practice

    if torch.cuda.is_available():
        torch.cuda.manual_seed(tetris_config.SEED + 100) # Different seed for testing
    else:
        torch.manual_seed(tetris_config.SEED + 100)
    random.seed(tetris_config.SEED + 100)
    np.random.seed(tetris_config.SEED + 100)

    env = Tetris(width=tetris_config.GAME_WIDTH, height=tetris_config.GAME_HEIGHT,
                 block_size=tetris_config.GAME_BLOCK_SIZE)

    # --- Agent Initialization and Loading ---
    agent_class = AGENT_REGISTRY.get(opt.agent_type)
    if not agent_class:
        print(f"Error: Agent type '{opt.agent_type}' not found in AGENT_REGISTRY.")
        return
    
    print(f"\n--- Testing Agent: {opt.agent_type.upper()} ---")
    agent = agent_class(state_size=tetris_config.STATE_SIZE, seed=tetris_config.SEED + 100)

    model_load_path = getattr(tetris_config, f"{opt.agent_type.upper()}_MODEL_PATH", None)
    if opt.agent_type == "random":
        print("Testing Random Agent. No model to load.")
    elif opt.agent_type == "ppo":
        actor_path = tetris_config.PPO_ACTOR_MODEL_PATH
        critic_path = tetris_config.PPO_CRITIC_MODEL_PATH
        if not os.path.exists(actor_path) or not os.path.exists(critic_path):
            print(f"ERROR: PPO model (actor or critic) not found. Train first.")
            return
        agent.load(actor_path, critic_path)
    elif model_load_path:
        if not os.path.exists(model_load_path):
            print(f"ERROR: Model for {opt.agent_type} not found at {model_load_path}. Train first.")
            return
        agent.load(model_load_path)
    else:
        print(f"Warning: No specific model path for {opt.agent_type} in config. Agent will be untrained.")
        # Allow testing untrained (e.g. initial random policy of a complex agent)

    # Set agent to evaluation mode (e.g., for DQN epsilon, PPO exploration flags)
    if hasattr(agent, 'qnetwork_local') and agent.qnetwork_local is not None: # DQN
        agent.qnetwork_local.eval()
    if hasattr(agent, 'qnetwork_target') and agent.qnetwork_target is not None: # DQN
        agent.qnetwork_target.eval()
    if hasattr(agent, 'policy_network') and agent.policy_network is not None: # REINFORCE, GA, ES
        agent.policy_network.eval()
    if hasattr(agent, 'actor') and agent.actor is not None: # PPO, A2C (if separate actor)
        agent.actor.eval()
    if hasattr(agent, 'critic') and agent.critic is not None: # PPO, A2C (if separate critic)
        agent.critic.eval()
    if hasattr(agent, 'network') and agent.network is not None: # A2C (if combined network)
        agent.network.eval()
    
    # For DQN, set epsilon to a very small value for testing (greedy policy)
    if opt.agent_type == "dqn" and hasattr(agent, 'epsilon'):
        agent.epsilon = 0.001 # Small exploration during testing

    # Video writer setup (from original Tetris test.py)
    # The Tetris render method produces a BGR image.
    # Output size needs to match what env.render() produces.
    # Original env.render() also adds an extra board for text.
    # Width: game_width * block_size + game_width * block_size / 2 (for text area)
    # Height: game_height * block_size
    render_width = tetris_config.GAME_WIDTH * tetris_config.GAME_BLOCK_SIZE + \
                   tetris_config.GAME_WIDTH * tetris_config.GAME_BLOCK_SIZE // 2
    render_height = tetris_config.GAME_HEIGHT * tetris_config.GAME_BLOCK_SIZE
    
    # Make sure GIF_DIR exists (it's actually for LunarLander, but video path is fine)
    video_path = os.path.join(tetris_config.MODEL_DIR, opt.output_video) # Save in models_tetris
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Using MJPG codec. Ensure it's available or try 'XVID' or 'mp4v'.
    # fourcc = cv2.VideoWriter_fourcc(*"MJPG") # For .avi typically
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # For .mp4
    out_video = cv2.VideoWriter(video_path, fourcc, opt.fps, (render_width, render_height))

    total_scores = []
    for i_game in range(opt.num_games):
        print(f"Starting Test Game: {i_game + 1}/{opt.num_games}")
        current_board_features = env.reset()
        if tetris_config.DEVICE.type == 'cuda':
            current_board_features = current_board_features.cuda()
        agent.reset() # Reset agent's game-specific state if any
        
        game_score = 0
        game_over = False
        
        max_test_pieces = tetris_config.MAX_PIECES_PER_EVAL_GAME # Limit game length for testing

        for piece_num in range(max_test_pieces):
            # Pass current_board_features and the env instance
            action_tuple, _ = agent.select_action(current_board_features, env, epsilon_override=0.0) # Greedy for testing
            
            # env.step returns reward for the single piece placement, and game_over status
            # The render method inside env.step will write to video if video object is passed
            reward, game_over = env.step(action_tuple, render=True, video=out_video)
            game_score += reward
            
            if game_over:
                # Render final game_over state once more if needed
                env.render(video=out_video) # Render the board one last time
                break
            
            current_board_features = env.get_state_properties(env.board)
            if tetris_config.DEVICE.type == 'cuda':
                current_board_features = current_board_features.cuda()
        
        print(f"  Game {i_game + 1} finished. Score: {game_score}, Tetrominoes: {env.tetrominoes}, Lines: {env.cleared_lines}")
        total_scores.append(game_score)

    out_video.release()
    print(f"\nTesting finished. Video saved to: {video_path}")
    if total_scores:
        print(f"Average score over {opt.num_games} games: {np.mean(total_scores):.2f}")
        print(f"Max score: {np.max(total_scores):.2f}, Min score: {np.min(total_scores):.2f}")

if __name__ == "__main__":
    test()