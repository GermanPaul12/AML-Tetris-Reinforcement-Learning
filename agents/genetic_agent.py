import os
import sys
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from src.tetris import Tetris
from .base_agent import BaseAgent, PolicyNetwork
import config as global_config

DEVICE = global_config.DEVICE

class GeneticAlgorithmController:
    def __init__(self, state_size,
        population_size=global_config.GA_POPULATION_SIZE,
        mutation_rate=global_config.GA_MUTATION_RATE,
        mutation_strength=global_config.GA_MUTATION_STRENGTH,
        crossover_rate=global_config.GA_CROSSOVER_RATE,
        tournament_size=global_config.GA_TOURNAMENT_SIZE,
        elitism_count=global_config.GA_ELITISM_COUNT,
        eval_games_per_individual=global_config.GA_EVAL_GAMES_PER_INDIVIDUAL,
        max_pieces_per_eval_game=global_config.GA_MAX_PIECES_PER_GA_EVAL_GAME
    ):
        print(f"GA Controller initialized. Population: {population_size}")
        self.state_size = state_size
        self.action_size = 1  # Tetris has a single action space (x, rotation)
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count

        self.eval_games_per_individual = eval_games_per_individual
        self.max_pieces_per_eval_game = max_pieces_per_eval_game

        self.population = self._initialize_population()
        self.best_individual_network = None  # Stores the best PolicyNetwork found
        self.best_fitness = -float("inf")

    def _initialize_population(self):
        population = []
        for i in range(self.population_size):
            individual_seed = global_config.SEED + i
            policy_net = PolicyNetwork(self.state_size, self.action_size, seed=individual_seed, fc1_units=global_config.GA_FC1_UNITS, fc2_units=global_config.GA_FC2_UNITS).to(DEVICE)
            population.append(policy_net)
        return population

    def _evaluate_fitness(self, individual_policy_net, eval_env_template: Tetris):
        """Evaluates fitness by playing a few full games."""
        total_rewards_for_individual = []
        individual_policy_net.eval()  # Set to evaluation mode

        for _ in range(self.eval_games_per_individual):
            # Create a fresh game environment for each evaluation game
            eval_env = Tetris()

            current_board_features = eval_env.reset()
            if DEVICE.type == "cuda": current_board_features = current_board_features.cuda()

            game_score = 0
            game_over = False
            pieces_this_game = 0

            while not game_over and pieces_this_game < self.max_pieces_per_eval_game:
                next_steps_dict = eval_env.get_next_states()

                possible_actions_tuples = list(next_steps_dict.keys())
                feature_tensors_for_next_steps = torch.stack(list(next_steps_dict.values())).to(DEVICE)

                with torch.no_grad(): action_scores = individual_policy_net(feature_tensors_for_next_steps)

                chosen_idx = torch.argmax(action_scores.squeeze()).item()
                chosen_action_tuple = possible_actions_tuples[chosen_idx]

                reward, game_over = eval_env.step(chosen_action_tuple, render=False)
                game_score += reward
                pieces_this_game += 1

                if not game_over:
                    current_board_features = eval_env.get_state_properties(eval_env.board)
                    if DEVICE.type == "cuda": current_board_features = current_board_features.cuda()

            total_rewards_for_individual.append(game_score)

        return (np.mean(total_rewards_for_individual) if total_rewards_for_individual else -float("inf"))

    def _selection(self, fitnesses):
        selected_parents = []
        for _ in range(self.population_size - self.elitism_count):
            tournament_indices = random.sample(range(len(self.population)), self.tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_index_in_tournament = np.argmax(tournament_fitnesses)
            selected_parents.append(self.population[tournament_indices[winner_index_in_tournament]])
        return selected_parents

    def _crossover(self, parent1_net: PolicyNetwork, parent2_net: PolicyNetwork):
        child1_net = PolicyNetwork(self.state_size, self.action_size, seed=random.randint(0, 1000000), fc1_units=global_config.GA_FC1_UNITS, fc2_units=global_config.GA_FC2_UNITS).to(DEVICE)
        child2_net = PolicyNetwork(self.state_size, self.action_size, seed=random.randint(0, 1000000), fc1_units=global_config.GA_FC1_UNITS, fc2_units=global_config.GA_FC2_UNITS).to(DEVICE)

        p1_weights = parent1_net.get_weights_flat()
        p2_weights = parent2_net.get_weights_flat()

        if random.random() < self.crossover_rate:
            # Averaging crossover
            child1_w = (p1_weights + p2_weights) / 2.0
            child2_w = (p1_weights + p2_weights) / 2.0

            child1_net.set_weights_flat(child1_w)
            child2_net.set_weights_flat(child2_w)
        else:
            child1_net.set_weights_flat(p1_weights)  # Clone
            child2_net.set_weights_flat(p2_weights)  # Clone
        return child1_net, child2_net

    def _mutate(self, individual_net: PolicyNetwork):
        weights = individual_net.get_weights_flat()
        for i in range(len(weights)):
            if random.random() < self.mutation_rate:
                noise = np.random.normal(0, self.mutation_strength)
                weights[i] += noise
        individual_net.set_weights_flat(weights)
        return individual_net

    def evolve_population(self, eval_env_template: Tetris):
        fitnesses = [self._evaluate_fitness(ind, eval_env_template) for ind in self.population]

        current_gen_best_idx = np.argmax(fitnesses)
        if fitnesses[current_gen_best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[current_gen_best_idx]
            self.best_individual_network = copy.deepcopy(self.population[current_gen_best_idx])
            self.best_individual_network.to(DEVICE)
            print(f"    GA: New best overall fitness: {self.best_fitness:.2f}")

        new_population = []
        sorted_indices = np.argsort(fitnesses)[::-1]  # Descending sort

        for i in range(self.elitism_count):
            elite_individual = copy.deepcopy(self.population[sorted_indices[i]])
            new_population.append(elite_individual.to(DEVICE))

        parents = self._selection(fitnesses)

        num_offspring_needed = self.population_size - self.elitism_count
        offspring_generated = 0
        parent_idx = 0
        while offspring_generated < num_offspring_needed:
            p1 = parents[parent_idx % len(parents)]
            p2 = parents[(parent_idx + 1 + random.randint(0, len(parents) - 2)) % len(parents)]
            parent_idx += 1

            child1, child2 = self._crossover(p1, p2)
            new_population.append(self._mutate(child1))
            offspring_generated += 1
            if offspring_generated < num_offspring_needed:
                new_population.append(self._mutate(child2))
                offspring_generated += 1

        self.population = new_population[: self.population_size]
        return np.mean(fitnesses), np.max(fitnesses)

    def get_best_policy_network(self):
        if self.best_individual_network is None and self.population:  # Fallback
            # This would require re-evaluating the current population if no best_individual_network is set yet.
            # It's better to ensure evolve_population is called at least once.
            print("Warning: Best individual network not yet identified. Returning first in population.")
            return self.population[0]
        return self.best_individual_network

    def save_best_individual(self, filename=None):
        path = filename if filename else global_config.GA_MODEL_PATH
        if self.best_individual_network:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.best_individual_network.state_dict(), path)
            print(f"Best GA Individual saved to: {path}")
        else:
            print("No best GA individual to save yet.")

    def load_best_individual(self, filename=None, state_size_override=None):
        path = filename if filename else global_config.GA_MODEL_PATH
        _state_size = (
            state_size_override if state_size_override is not None else self.state_size
        )
        if not _state_size:
            print("Error: Cannot load GA model without state_size.")
            return False

        if os.path.exists(path):
            loaded_net = PolicyNetwork(_state_size, self.action_size, seed=0, fc1_units=global_config.GA_FC1_UNITS, fc2_units=global_config.GA_FC2_UNITS).to(DEVICE)
            loaded_net.load_state_dict(torch.load(path, map_location=DEVICE))
            loaded_net.eval()
            self.best_individual_network = loaded_net
            # self.best_fitness should be re-evaluated or set to a known value if stored elsewhere
            print(f"Best GA Individual loaded from {path}. Fitness needs re-evaluation.")
            return True
        else:
            print(f"Error: GA model file not found at {path}")
            return False


class GeneticAgent(BaseAgent):
    """Agent wrapper that uses the best policy found by GAController."""

    def __init__(self, state_size, seed=0, policy_network_instance: PolicyNetwork = None):
        super().__init__(state_size)
        self.policy_network = policy_network_instance
        if self.policy_network is None:
            print("GeneticAgent initialized without a policy network. Will use a default one (likely untrained).")
            # Create a default one, but it won't be useful until loaded or set
            self.policy_network = PolicyNetwork(state_size, action_size=1, seed=seed, fc1_units=global_config.GA_FC1_UNITS, fc2_units=global_config.GA_FC2_UNITS).to(DEVICE)
        self.policy_network.eval()  # Ensure it's in eval mode

    def select_action(self, current_board_features: torch.Tensor, tetris_game_instance: Tetris, epsilon_override=None):
        next_steps_dict = tetris_game_instance.get_next_states()
        if not next_steps_dict:
            return (tetris_game_instance.width // 2, 0), {}  # Fallback

        possible_actions_tuples = list(next_steps_dict.keys())
        feature_tensors_for_next_steps = torch.stack(list(next_steps_dict.values())).to(DEVICE)

        with torch.no_grad(): action_scores = self.policy_network(feature_tensors_for_next_steps)

        chosen_idx = torch.argmax(action_scores.squeeze()).item()
        chosen_action_tuple = possible_actions_tuples[chosen_idx]

        # Aux_info can be minimal for GA during inference
        return chosen_action_tuple, {"features_after_chosen_action": feature_tensors_for_next_steps[chosen_idx]}

    def set_policy_network(self, policy_net: PolicyNetwork):
        self.policy_network = policy_net
        if self.policy_network:
            self.policy_network.to(DEVICE).eval()

    def save(self, filepath=None):  # Saves the current policy network of this agent instance
        path = (filepath if filepath else global_config.GA_MODEL_PATH)  # Usually save controller's best
        if self.policy_network:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.policy_network.state_dict(), path)
            print(f"GeneticAgent's current policy saved to: {path}")
        else:
            print("GeneticAgent has no policy network to save.")

    def load(self, filepath=None):  # Loads into this agent instance's policy network
        path = filepath if filepath else global_config.GA_MODEL_PATH
        if os.path.exists(path):
            # Re-initialize network if it doesn't exist, using self.state_size
            if self.policy_network is None:
                self.policy_network = PolicyNetwork(self.state_size, action_size=1, seed=0, fc1_units=global_config.GA_FC1_UNITS, fc2_units=global_config.GA_FC2_UNITS).to(DEVICE)
            self.policy_network.load_state_dict(torch.load(path, map_location=DEVICE))
            self.policy_network.eval()
            print(f"GeneticAgent policy loaded from: {path}")
        else:
            print(f"Error: GA model file not found at {path} for GeneticAgent instance.")
            raise FileNotFoundError(f"GA model not found for GeneticAgent: {path}")
