# tetris_rl_agents/agents/__init__.py
from .base_agent import BaseAgent
from .random_agent import RandomAgent
from .dqn_agent import DQNAgent
from .genetic_agent import GeneticAgent, GeneticAlgorithmController, PolicyNetwork as GAPolicyNetwork
from .reinforce_agent import REINFORCEAgent, PolicyNetworkREINFORCE
from .a2c_agent import A2CAgent #, ActorCriticNetwork as A2CNetwork (if defined)
from .ppo_agent import PPOAgent #, ActorPPO, CriticPPO (if defined)
from .es_agent import ESAgent, PolicyNetworkES

AGENT_REGISTRY = {
    "random": RandomAgent,
    "dqn": DQNAgent,
    "genetic": GeneticAgent,
    "reinforce": REINFORCEAgent,
    "a2c": A2CAgent,
    "ppo": PPOAgent,
    "es": ESAgent,
}