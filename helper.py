import os
import re
import torch
import config
import random
import numpy as np

#########################################################################
# Helper Functions for Agent File Management, Score Parsing and Seeding #
#########################################################################

def get_agent_file_prefix(agent_type_str:str, is_actor:bool=False, is_critic:bool=False) -> str:
    """Generates a standardized file prefix for saving/loading agent models."""
    processed_agent_type = agent_type_str.replace("_", "-")
    if agent_type_str == "ppo":
        if is_actor: return "ppo-actor"
        elif is_critic: return "ppo-critic"
        else: return "ppo-model"
    return processed_agent_type

def parse_score_from_filename(filename_basename:str, expected_prefix:str) -> (int | None):
    """Extracts the score from a filename that matches the expected pattern."""
    pattern = re.compile(f"^{re.escape(expected_prefix)}_score_(\\d+)\\.pth$")
    match = pattern.match(filename_basename)
    if match:
        try: return int(match.group(1))
        except ValueError: return None
    return None

def find_best_existing_score(agent_prefix:str, model_dir:str) -> int:
    """Finds the highest score from existing model files in the specified directory."""
    max_score = -1
    if not os.path.isdir(model_dir):
        try: os.makedirs(model_dir, exist_ok=True)
        except OSError:
            print(f"Warning: Model directory {model_dir} does not exist and could not be created.")
            return max_score
    for filename in os.listdir(model_dir):
        score = parse_score_from_filename(filename, agent_prefix)
        if score is not None and score > max_score:
            max_score = score
    return max_score

def setup_seeds() -> None:
    """Sets random seeds for reproducibility across numpy, torch, and random."""
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    print(f"Seeds set to: {config.SEED}")