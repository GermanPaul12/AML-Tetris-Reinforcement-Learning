# tetris_rl_agents/config.py
import os
import torch

# --- General Game & Project Configuration ---
GAME_WIDTH = 10
GAME_HEIGHT = 20
GAME_BLOCK_SIZE = 30
STATE_SIZE = 4  # [lines_cleared_in_move, holes_after_move, bumpiness_after_move, height_after_move]
SEED = 123
PROJECT_ROOT = "."

# --- Test Configuration (for test.py) ---
NUM_TEST_RUNS_GIF = 1  # Number of full games to record in one GIF
RENDER_MODE_TEST = "rgb_array"  # For GIF: "rgb_array", for viewing: "human"
GIF_FPS = 300

# --- Evaluation Configuration (for evaluate.py) ---
NUM_EVAL_GAMES = 20  # Number of full games for final evaluation
MAX_PIECES_PER_EVAL_GAME = 1000000000
RENDER_MODE_EVAL = "None"  # None for faster, "human" for viewing

# --- Device Configuration ---
DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else 
    "cpu"
)

# --- Agent Types ---
AGENT_TYPES = ["random", "dqn", "genetic", "reinforce", "a2c", "ppo", "es"]

# --- Model Paths & Directories ---
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

def ensure_model_dir_exists():
    if not os.path.exists(MODEL_DIR):
        try:
            os.makedirs(MODEL_DIR)
            print(f"Model directory created: {MODEL_DIR}")
        except OSError as e:
            print(f"Error creating model directory {MODEL_DIR}: {e}")

DQN_MODEL_FILENAME = "dqn_tetris.pth"
GA_MODEL_FILENAME = "ga_best_tetris.pth"
REINFORCE_MODEL_FILENAME = "reinforce_tetris.pth"
A2C_MODEL_FILENAME = "a2c_tetris.pth"
PPO_ACTOR_MODEL_FILENAME = "ppo_actor_tetris.pth"
PPO_CRITIC_MODEL_FILENAME = "ppo_critic_tetris.pth"
ES_MODEL_FILENAME = "es_tetris.pth"
EVALUATION_CSV_FILENAME = "evaluation_summary_tetris.csv"

DQN_MODEL_PATH = os.path.join(MODEL_DIR, DQN_MODEL_FILENAME)
GA_MODEL_PATH = os.path.join(MODEL_DIR, GA_MODEL_FILENAME)
REINFORCE_MODEL_PATH = os.path.join(MODEL_DIR, REINFORCE_MODEL_FILENAME)
A2C_MODEL_PATH = os.path.join(MODEL_DIR, A2C_MODEL_FILENAME)
PPO_ACTOR_MODEL_PATH = os.path.join(MODEL_DIR, PPO_ACTOR_MODEL_FILENAME)
PPO_CRITIC_MODEL_PATH = os.path.join(MODEL_DIR, PPO_CRITIC_MODEL_FILENAME)
ES_MODEL_PATH = os.path.join(MODEL_DIR, ES_MODEL_FILENAME)
EVALUATION_CSV_PATH = os.path.join(MODEL_DIR, EVALUATION_CSV_FILENAME)

# --- Training Configuration ---
FORCE_RETRAIN_ALL = False
FORCE_RETRAIN_DQN = False or FORCE_RETRAIN_ALL
FORCE_RETRAIN_GA = False or FORCE_RETRAIN_ALL
FORCE_RETRAIN_REINFORCE = False or FORCE_RETRAIN_ALL
FORCE_RETRAIN_A2C = False or FORCE_RETRAIN_ALL
FORCE_RETRAIN_PPO = False or FORCE_RETRAIN_ALL
FORCE_RETRAIN_ES = False or FORCE_RETRAIN_ALL

# --- General training parameters (can be overridden per agent) ---
MAX_EPOCHS = 5000
MAX_EPOCHS_OR_PIECES = 50000
PRINT_EVERY_EPOCHS = 100
SCORE_TARGET = 1000000
EARLY_STOPPING_TARGET_SCORE = 1000000

# === Double Q-Network (DQN) ===
DQN_NUM_EPOCHS = 3000 
DQN_MAX_T_PER_GAME_EVAL = 10000
DQN_PRINT_EVERY_GAMES = 10
DQN_TARGET_GAME_SCORE = 5000000

# DQN Hyperparameters
DQN_BUFFER_SIZE = 30000
DQN_BATCH_SIZE = 512
DQN_GAMMA = 0.99
DQN_LR = 1e-3
DQN_UPDATE_EVERY = 1  # For Tetris, learn after every piece placement
DQN_TARGET_UPDATE_EVERY = 1000  # How many learning steps before target net update
DQN_EPSILON_START = 1.0
DQN_EPSILON_MIN = 1e-3
DQN_EPSILON_DECAY_EPOCHS = 2000  # Epochs over which epsilon decays

# DQN Network Architecture
DQN_FC1_UNITS = 64
DQN_FC2_UNITS = 64

# === Genetic Algorithm (GA) ===
GA_N_GENERATIONS = 200
GA_POPULATION_SIZE = 50
GA_EVAL_GAMES_PER_INDIVIDUAL = 1
GA_MAX_PIECES_PER_GA_EVAL_GAME = 100000000000
GA_SAVE_EVERY_N_GENERATIONS = 10

# GA Hyperparameters
GA_MUTATION_RATE = 0.1
GA_MUTATION_STRENGTH = 0.15
GA_CROSSOVER_RATE = 0.7
GA_TOURNAMENT_SIZE = 5
GA_ELITISM_COUNT = 2

# GA Training Parameters
GA_FC1_UNITS = 32
GA_FC2_UNITS = 32

# === Evolutional Strategies (ES) ===
ES_N_GENERATIONS = 300
ES_POPULATION_SIZE = 50
ES_SIGMA = 0.1
ES_LEARNING_RATE = 0.005
ES_EVAL_GAMES_PER_PARAM = 1
ES_MAX_PIECES_PER_ES_EVAL_GAME = 10000000
ES_PRINT_EVERY_GENS = 1
ES_TARGET_GAME_SCORE = 1000000

ES_FC1_UNITS = 32
ES_FC2_UNITS = 32

# === REINFORCE ===
REINFORCE_TRAIN_GAMES = 50000  # Number of full games
REINFORCE_MAX_PIECES_PER_GAME = 1000000000
REINFORCE_PRINT_EVERY_GAMES = 10
REINFORCE_TARGET_GAME_SCORE = 1000000

REINFORCE_LEARNING_RATE = 1e-4
REINFORCE_GAMMA = 0.99
REINFORCE_FC1_UNITS = 128
REINFORCE_FC2_UNITS = 128

# === A2C ===
A2C_TRAIN_GAMES = 50000
A2C_MAX_PIECES_PER_GAME = 10000000000
A2C_PRINT_EVERY_GAMES = 10
A2C_TARGET_GAME_SCORE = 1000000

A2C_LEARNING_RATE = 7e-4
A2C_GAMMA = 0.99
A2C_ENTROPY_COEFF = 0.01
A2C_VALUE_LOSS_COEFF = 0.5
A2C_FC1_UNITS = 64  # Shared layers
A2C_FC2_UNITS = 64

# === PPO ===
PPO_TOTAL_PIECES = 1000000000  # Total piece placements
PPO_TRAIN_GAMES = 50000  # Number of full games for training
PPO_UPDATE_HORIZON = 1024  # Pieces collected before update
PPO_EPOCHS_PER_UPDATE = 4  # SGD epochs over collected data
PPO_BATCH_SIZE = 64
PPO_PRINT_EVERY_N_UPDATES = 5
PPO_TARGET_GAME_SCORE = 2000000
PPO_MAX_POTENTIAL_ACTIONS = 50

PPO_ACTOR_LR = 3e-4
PPO_CRITIC_LR = 1e-3
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_EPSILON = 0.2
PPO_ENTROPY_COEFF = 0.01
PPO_VALUE_LOSS_COEFF = 0.5
PPO_ACTOR_FC1 = 64  # Actor net: state_after_action -> score
PPO_ACTOR_FC2 = 64
PPO_CRITIC_FC1 = 64  # Critic net: state_before_action -> value
PPO_CRITIC_FC2 = 64
