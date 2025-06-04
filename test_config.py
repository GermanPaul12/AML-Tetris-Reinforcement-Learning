# tetris_rl_agents/config_test_suite.py
# This is a configuration for RAPID TESTING & DEBUGGING, not for optimal performance.
import os
import torch

# --- General Game & Project Configuration (mostly same as main config) ---
GAME_WIDTH = 10
GAME_HEIGHT = 20
GAME_BLOCK_SIZE = 30
STATE_SIZE = 4
SEED = 42 # Use a different seed for test suite to avoid interfering with main runs if needed
PROJECT_ROOT = "."
MODEL_DIR = os.path.join(PROJECT_ROOT, "models_test_suite") # Separate dir for test models

# --- Agent Types (can test a subset if desired) ---
AGENT_TYPES = ["random", "dqn_original", "dqn", "genetic", "reinforce", "a2c", "ppo", "es"]

# --- Model Paths (point to test_suite directory) ---
ORIGINAL_DQN_MODEL_FILENAME = "dqn_tetris_original_test.pth"
ORIGINAL_DQN_MODEL_PATH = os.path.join(MODEL_DIR, ORIGINAL_DQN_MODEL_FILENAME)
# ... (Define all other model paths similarly, pointing to MODEL_DIR)
DQN_MODEL_FILENAME = "dqn_tetris_test.pth"
DQN_MODEL_PATH = os.path.join(MODEL_DIR, DQN_MODEL_FILENAME)
GA_MODEL_FILENAME = "ga_best_tetris_test.pth"
GA_MODEL_PATH = os.path.join(MODEL_DIR, GA_MODEL_FILENAME)
REINFORCE_MODEL_FILENAME = "reinforce_tetris_test.pth"
REINFORCE_MODEL_PATH = os.path.join(MODEL_DIR, REINFORCE_MODEL_FILENAME)
A2C_MODEL_FILENAME = "a2c_tetris_test.pth"
A2C_MODEL_PATH = os.path.join(MODEL_DIR, A2C_MODEL_FILENAME)
PPO_ACTOR_MODEL_FILENAME = "ppo_actor_tetris_test.pth"
PPO_CRITIC_MODEL_FILENAME = "ppo_critic_tetris_test.pth"
PPO_ACTOR_MODEL_PATH = os.path.join(MODEL_DIR, PPO_ACTOR_MODEL_FILENAME)
PPO_CRITIC_MODEL_PATH = os.path.join(MODEL_DIR, PPO_CRITIC_MODEL_FILENAME)
ES_MODEL_FILENAME = "es_tetris_test.pth"
ES_MODEL_PATH = os.path.join(MODEL_DIR, ES_MODEL_FILENAME)

EVALUATION_CSV_FILENAME = "evaluation_summary_tetris_test.csv"
EVALUATION_CSV_PATH = os.path.join(MODEL_DIR, EVALUATION_CSV_FILENAME)


# --- General Training Configuration for Test Suite ---
# These will be short runs!
MAX_EPOCHS_OR_PIECES = 500 # VERY LOW - For step-based agents (e.g. ~2-3 games)
SCORE_TARGET = 100         # Low target, unlikely to be hit with these settings

# === DQN / OriginalDQN ===
DQN_NUM_EPOCHS = 500           # Number of piece placements (original was 3000 for full)
DQN_BUFFER_SIZE = 1000         # Small buffer
DQN_BATCH_SIZE = 32            # Small batch
DQN_LR = 5e-3                  # Higher LR for faster (but maybe unstable) initial changes
DQN_GAMMA = 0.95               # Slightly less future focus for faster initial signal
DQN_UPDATE_EVERY = 1
DQN_TARGET_UPDATE_EVERY = 20   # Frequent target updates
DQN_EPSILON_START = 1.0
DQN_EPSILON_MIN = 0.1          # Don't decay epsilon too much
DQN_EPSILON_DECAY_EPOCHS = 200 # Fast decay

DQN_FC1_UNITS = 16             # Smaller network
DQN_FC2_UNITS = 16

# === Genetic Algorithm (GA) ===
GA_N_GENERATIONS = 10           # Very few generations
GA_POPULATION_SIZE = 10        # Small population
GA_EVAL_GAMES_PER_INDIVIDUAL = 1
GA_MAX_PIECES_PER_GA_EVAL_GAME = 100 # Short eval games
GA_SAVE_EVERY_N_GENERATIONS = 2

GA_MUTATION_RATE = 0.2         # Higher mutation for more exploration
GA_MUTATION_STRENGTH = 0.2
GA_CROSSOVER_RATE = 0.8
GA_TOURNAMENT_SIZE = 3
GA_ELITISM_COUNT = 1

GA_FC1_UNITS = 16
GA_FC2_UNITS = 16

# === Evolutionary Strategies (ES) ===
ES_N_GENERATIONS = 20
ES_POPULATION_SIZE = 10
ES_SIGMA = 0.2                 # Larger noise
ES_LEARNING_RATE = 0.05        # Higher LR
ES_EVAL_GAMES_PER_PARAM = 1
ES_MAX_PIECES_PER_ES_EVAL_GAME = 100
ES_PRINT_EVERY_GENS = 1
ES_TARGET_GAME_SCORE = 100

ES_FC1_UNITS = 16
ES_FC2_UNITS = 16

# === REINFORCE (with Baseline) ===
REINFORCE_TRAIN_GAMES = 20     # Very few games
REINFORCE_MAX_PIECES_PER_GAME = 100
REINFORCE_LEARNING_RATE = 1e-3 # Higher LR for policy and value
REINFORCE_GAMMA = 0.95
REINFORCE_FC1_UNITS = 16
REINFORCE_FC2_UNITS = 16

# === A2C ===
A2C_TRAIN_GAMES = 20
A2C_MAX_PIECES_PER_GAME = 100
A2C_LEARNING_RATE = 1e-3       # Higher LR
A2C_GAMMA = 0.95
A2C_ENTROPY_COEFF = 0.05       # Higher entropy for more exploration initially
A2C_VALUE_LOSS_COEFF = 0.5
A2C_FC1_UNITS = 16
A2C_FC2_UNITS = 16

# === PPO ===
PPO_TOTAL_PIECES = 1000        # Short total interaction
PPO_UPDATE_HORIZON = 128       # Small horizon, frequent updates
PPO_EPOCHS_PER_UPDATE = 2      # Few SGD epochs
PPO_BATCH_SIZE = 32            # Small batch
PPO_MAX_POTENTIAL_ACTIONS = 50 # Keep this reasonable, doesn't affect speed as much as learning

PPO_ACTOR_LR = 5e-4            # Higher LRs
PPO_CRITIC_LR = 1e-3
PPO_GAMMA = 0.95
PPO_GAE_LAMBDA = 0.9
PPO_CLIP_EPSILON = 0.2         # Standard clip
PPO_ENTROPY_COEFF = 0.02       # More exploration
PPO_VALUE_LOSS_COEFF = 0.5
PPO_ACTOR_FC1 = 16
PPO_ACTOR_FC2 = 16
PPO_CRITIC_FC1 = 16
PPO_CRITIC_FC2 = 16

# --- Test & Evaluation Configuration (can be kept short for test suite) ---
NUM_TEST_RUNS_GIF = 1
RENDER_MODE_TEST = "rgb_array" # or "human" if you want to watch one run
GIF_FPS = 10 # Slower GIF for easier inspection

NUM_EVAL_GAMES = 2 # Very few eval games
MAX_PIECES_PER_EVAL_GAME = 100
RENDER_MODE_EVAL = None

# --- Device Configuration ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ensure_model_dir_exists(): # Call this to create the test_suite model dir
    if not os.path.exists(MODEL_DIR):
        try:
            os.makedirs(MODEL_DIR)
            print(f"Test suite model directory created: {MODEL_DIR}")
        except OSError as e:
            print(f"Error creating test suite model directory {MODEL_DIR}: {e}")