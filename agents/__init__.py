# tetris_rl_agents/agents/__init__.py
from .random_agent import RandomAgent
from .dqn_agent import DQNAgent
from .genetic_agent import GeneticAgent
from .reinforce_agent import REINFORCEAgent
from .a2c_agent import A2CAgent
from .ppo_agent import PPOAgent
from .es_agent import ESAgent

AGENT_REGISTRY = {
    "random": RandomAgent,
    "dqn": DQNAgent,
    "genetic": GeneticAgent,
    "reinforce": REINFORCEAgent,
    "a2c": A2CAgent,
    "ppo": PPOAgent,
    "es": ESAgent,
}
