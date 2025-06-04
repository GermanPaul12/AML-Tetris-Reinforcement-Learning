"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

from random import random, randint, sample
from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque


def get_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--initial_epsilon", type=float, default=1, help="Initial epsilon for epsilon-greedy exploration")
    parser.add_argument("--final_epsilon", type=float, default=1e-3, help="Final epsilon for epsilon-greedy exploration")
    parser.add_argument("--num_decay_epochs", type=float, default=2000, help="Number of epochs for epsilon decay")
    parser.add_argument("--num_epochs", type=int, default=3000, help="Total number of training epochs")
    parser.add_argument("--save_interval", type=int, default=1000, help="Number of epoches between saving the model")
    parser.add_argument("--replay_memory_size", type=int, default=30000, help="Number of epoches between testing phases")
    parser.add_argument("--saved_path", type=str, default="models", help="Path to save the model")
    parser.add_argument("--render", type=bool, default=False, help="Render the game during training")

    args = parser.parse_args()
    return args

def get_predictions(model, next_states):
    """Get predictions from the model for the next states."""
    model.eval()
    with torch.no_grad():
        predictions = model(next_states)[:, 0]
    model.train()
    return predictions

def train(opt):
    print("Training Tetris with Deep Q Network")
    
    cuda_available = True if torch.cuda.is_available() else False
    print(f"Using {'GPU' if cuda_available else 'CPU'} for training.")

    torch.cuda.manual_seed(123) if cuda_available else torch.manual_seed(123) 
    
    # Initialisiere die Umgebung und das Modell
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    state = env.reset()
    
    if cuda_available:
        model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    
    while epoch < opt.num_epochs:
        
        # Setzen des Epsilon-Werts für die Exploration
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)

        # Erhalten der möglichen nächsten Zustände und Aktionen
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if cuda_available: next_states = next_states.cuda()
        
        # Berechnen der Vorhersagen für die nächsten Zustände
        predictions = get_predictions(model, next_states)
        
        # Wähle eine Aktion basierend auf Epsilon-Greedy-Strategie
        if random() <= epsilon:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        # Führe die gewählte Aktion aus
        next_state, action = next_states[index, :], next_actions[index]
        reward, done = env.step(action, render=opt.render)
        if cuda_available: next_state = next_state.cuda()
        
        # Speichere den Übergang im Replay-Speicher
        replay_memory.append([state, reward, next_state, done])
        
        # Wenn das Spiel beendet ist, aktualisiere den Zustand und speichere die Endergebnisse.
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            if cuda_available: state = state.cuda()
        else:
            state = next_state
            continue
        
        # Wenn der Replay-Speicher zu klein ist, überspringe das Training
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        
        # Trainiere das Modell
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        if cuda_available:
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()
        
        # Berechne die Vorhersagen für die nächsten Zustände
        next_prediction_batch = get_predictions(model, next_state_batch)

        # Berechne die Zielwerte für das Training
        y_batch = torch.cat(tuple(reward if done else reward + opt.gamma * prediction 
                                  for reward, done, prediction in zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        # Trainiere das Modell mit dem aktuellen Zustand und den Zielwerten
        optimizer.zero_grad()
        loss = criterion(model(state_batch), y_batch)
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}/{opt.num_epochs}, Epsilon: {epsilon}, Score: {final_score}, Tetrominoes {final_tetrominoes}, Cleared lines: {final_cleared_lines}")

        if (epoch > 0 and epoch % opt.save_interval == 0):
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))
        
        if final_score >= 1000000:
            print("You win! Saving the model...")
            break

    torch.save(model, "{}/tetris".format(opt.saved_path))
    print("Training completed. Model saved to {}/tetris".format(opt.saved_path))

if __name__ == "__main__":
    train(get_args())
