# tetris_rl_agents/train.py
import argparse
import os
import random
import copy # For deep copying game state

import numpy as np
import torch

import config as tetris_config
#import test_config as tetris_config
from agents import AGENT_REGISTRY
# Specific agent imports to check instance types if needed for very specific logic
from agents.dqn_agent import DQNAgent
from agents.original_dqn_agent import OriginalDQNAgent
from agents.reinforce_agent import REINFORCEAgent # For learn_episode call
from agents.a2c_agent import A2CAgent # For on-policy nature
from agents.ppo_agent import PPOAgent # For on-policy nature and horizon learning

from src.tetris import Tetris

def get_args():
    parser = argparse.ArgumentParser(
        """Train Reinforcement Learning Agents to play Tetris""")
    
    parser.add_argument("--agent_type", type=str, default="dqn_original", # Default to original for comparison
                        choices=list(tetris_config.AGENT_TYPES),
                        help="Type of agent to train.")
    parser.add_argument("--num_total_pieces", type=int, default=None,
                        help="Total number of pieces to play/learn from (overrides agent's config if set).")
    parser.add_argument("--render_game", action="store_true", help="Render the game during training.")
    parser.add_argument("--save_interval_pieces", type=int, default=5000,
                        help="Save model every N pieces played.")
    
    args = parser.parse_args()
    return args

def train():
    opt = get_args()
    tetris_config.ensure_model_dir_exists()

    # Setup seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tetris_config.SEED)
    else:
        torch.manual_seed(tetris_config.SEED)
    random.seed(tetris_config.SEED)
    np.random.seed(tetris_config.SEED)

    env = Tetris(width=tetris_config.GAME_WIDTH, height=tetris_config.GAME_HEIGHT,
                 block_size=tetris_config.GAME_BLOCK_SIZE)

    # --- Agent Initialization ---
    agent_class = AGENT_REGISTRY.get(opt.agent_type)
    if not agent_class:
        print(f"Error: Agent type '{opt.agent_type}' not found.")
        return
    print(f"\n--- Training Agent: {opt.agent_type.upper()} ---")
    agent = agent_class(state_size=tetris_config.STATE_SIZE, seed=tetris_config.SEED)
    
    # --- Determine Training Duration (Number of Piece Placements) ---
    total_pieces_to_play = opt.num_total_pieces
    if total_pieces_to_play is None: # Get from config if not overridden
        if opt.agent_type == "dqn" or opt.agent_type == "dqn_original": # Use DQN_NUM_EPOCHS
            total_pieces_to_play = tetris_config.DQN_NUM_EPOCHS
        elif opt.agent_type == "ppo":
            total_pieces_to_play = tetris_config.PPO_TOTAL_PIECES
        elif opt.agent_type in ["reinforce", "a2c"]: # Estimate pieces based on games
            # These agents learn per game or per step within a game context
            # We still need a piece limit for the overall loop
            game_based_agent_train_games = getattr(tetris_config, f"{opt.agent_type.upper()}_TRAIN_GAMES", 5000)
            avg_pieces_per_game_estimate = 150 # A rough estimate
            total_pieces_to_play = game_based_agent_train_games * avg_pieces_per_game_estimate
            print(f"Training {opt.agent_type} for approx. {game_based_agent_train_games} games (piece limit: {total_pieces_to_play})")
        elif opt.agent_type in ["genetic", "es"]:
            print(f"ERROR: Agent type '{opt.agent_type}' is evolutionary. Use 'train_evolutionary.py'.")
            return
        else: # Fallback for other unknown step-based agents
            total_pieces_to_play = tetris_config.MAX_EPOCHS_OR_PIECES
    
    print(f"Starting training for {total_pieces_to_play} piece placements...")

    # --- Training Loop (Mimicking Original Tetris DQN Flow) ---
    
    # s_t_features: Features of the board *before* the current piece placement decision.
    s_t_features = env.reset() # Initial board state features
    if tetris_config.DEVICE.type == 'cuda':
        s_t_features = s_t_features.cuda()
    
    pieces_played_count = 0
    games_played_count = 0
    current_game_score = 0
    # env.tetrominoes and env.cleared_lines are part of env state

    # PPO specific: needs to know the last observation if horizon ends mid-episode
    last_s_t_plus_1_features_for_ppo = None 

    while pieces_played_count < total_pieces_to_play:
        # 1. Agent selects action based on s_t_features and current env state (to get next_states)
        #    `aux_info` for DQN-like agents should contain `s_prime_chosen_features`
        #    `aux_info` for Policy Gradient agents should contain `log_prob`, etc.
        action_tuple, aux_info = agent.select_action(s_t_features, env)
        
        # 2. Environment takes a step
        #    `reward` is for placing the current piece.
        #    `game_over` is True if this placement ends the game.
        reward, game_over = env.step(action_tuple, render=opt.render_game)
        
        pieces_played_count += 1
        current_game_score += reward

        # 3. Get S'_{actual}: Features of the board *after* the chosen piece has landed
        #    AND a *new piece* has appeared at the top (if game not over).
        #    If game_over, this is the final board state's features.
        s_prime_actual_features = env.get_state_properties(env.board)
        if tetris_config.DEVICE.type == 'cuda':
            s_prime_actual_features = s_prime_actual_features.cuda()
        
        last_s_t_plus_1_features_for_ppo = s_prime_actual_features # For PPO's potential final value est.

        # 4. Agent learns
        #    For DQN-like agents, `game_instance_at_s_prime` is crucial. It's a copy of the
        #    environment in state S'_{actual}, allowing the agent to call `get_next_states()`
        #    on it to find max_a' Q(S'_{actual}, a').
        game_instance_at_s_prime = None
        if not game_over:
            if isinstance(agent, DQNAgent) or isinstance(agent, OriginalDQNAgent) or \
               isinstance(agent, A2CAgent) or isinstance(agent, PPOAgent): # Agents needing to look ahead from S'_{actual}
                game_instance_at_s_prime = copy.deepcopy(env)
        
        agent.learn(
            state_features=s_t_features,  # S_t (board before action)
            action_tuple=action_tuple,    # A_t
            reward=reward,                # R_{t+1}
            next_state_features=s_prime_actual_features, # S'_{actual} (board after action & new piece)
            done=game_over,
            game_instance_at_s=None, # Not typically needed if select_action uses current env
            game_instance_at_s_prime=game_instance_at_s_prime, # Env state at S'_{actual}
            aux_info=aux_info             # Contains s_prime_chosen_features for DQN, log_probs for PG etc.
        )
        
        # 5. Update current state for next iteration
        s_t_features = s_prime_actual_features

        # 6. Handle game over
        if game_over:
            games_played_count += 1
            print(
                f"Pieces: {pieces_played_count}/{total_pieces_to_play} | "
                f"Game: {games_played_count} | Score: {current_game_score} | "
                f"Tetrominoes: {env.tetrominoes} | Lines: {env.cleared_lines} "
                f"{f'| Epsilon: {agent.epsilon:.4f}' if hasattr(agent, 'epsilon') else ''}"
            )
            
            if isinstance(agent, REINFORCEAgent): # REINFORCE learns at end of game
                agent.learn_episode()

            agent.reset() # Agent's own reset logic (e.g. clear episode buffers)
            
            s_t_features = env.reset() # Reset environment for new game
            if tetris_config.DEVICE.type == 'cuda':
                s_t_features = s_t_features.cuda()
            current_game_score = 0
            # env.tetrominoes and env.cleared_lines are reset by env.reset()

        # 7. PPO specific: potentially learn from memory if horizon is full
        if isinstance(agent, PPOAgent):
            if len(agent.memory_chosen_action_indices) >= agent.update_horizon:
                # If game ended exactly at horizon, last_s_t_plus_1 is S_T (terminal)
                # If game continues, last_s_t_plus_1 is the current s_t_features (which is S'_{actual} of last step)
                final_val_obs_for_ppo = None if game_over else last_s_t_plus_1_features_for_ppo
                agent.learn_from_memory(final_val_obs_for_ppo)


        # 8. Save model periodically
        if pieces_played_count > 0 and pieces_played_count % opt.save_interval_pieces == 0:
            save_path_prefix = opt.agent_type # For PPO
            
            model_save_path = getattr(tetris_config, f"{opt.agent_type.upper()}_MODEL_PATH", None)
            if opt.agent_type == "dqn_original": # Handle specific naming for original
                model_save_path = getattr(tetris_config, "ORIGINAL_DQN_MODEL_PATH", 
                                           os.path.join(tetris_config.MODEL_DIR, "dqn_tetris_original_impl.pth"))

            if opt.agent_type == "ppo":
                actor_path = tetris_config.PPO_ACTOR_MODEL_PATH
                critic_path = tetris_config.PPO_CRITIC_MODEL_PATH
                agent.save(actor_path, critic_path)
            elif model_save_path:
                agent.save(model_save_path)
            else: # Fallback generic save path
                agent.save(os.path.join(tetris_config.MODEL_DIR, f"{opt.agent_type}_tetris_pieces_{pieces_played_count}.pth"))
        
        # Check for overall target score if applicable (more relevant for full game metrics)
        # if current_game_score >= tetris_config.SCORE_TARGET and game_over :
        #     print(f"Target game score of {tetris_config.SCORE_TARGET} reached in game {games_played_count}!")
            # Could add logic to stop all training here if desired

    print("\nTraining finished.")
    # Final save
    final_model_save_path = getattr(tetris_config, f"{opt.agent_type.upper()}_MODEL_PATH", None)
    if opt.agent_type == "dqn_original":
         final_model_save_path = getattr(tetris_config, "ORIGINAL_DQN_MODEL_PATH",
                                    os.path.join(tetris_config.MODEL_DIR, "dqn_tetris_original_impl_final.pth"))
    if opt.agent_type == "ppo":
        agent.save(tetris_config.PPO_ACTOR_MODEL_PATH, tetris_config.PPO_CRITIC_MODEL_PATH)
    elif final_model_save_path:
        agent.save(final_model_save_path)
    else:
        agent.save(os.path.join(tetris_config.MODEL_DIR, f"{opt.agent_type}_tetris_final.pth"))

if __name__ == "__main__":
    train()