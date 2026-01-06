"""
Smart Waste Collection Routing System
Agent-Based Evolutionary Approach with Expert Rules
For: Intro to AI Class
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict, Any, Optional
import itertools
from dataclasses import dataclass
import copy
from collections import defaultdict
from scipy.spatial import Voronoi
import networkx as nx
import pandas as pd

# ============================================================================
# DATA GENERATION MODULE (Week 1)
# ============================================================================

class WasteBinGenerator:
    """Generate synthetic city grid with waste bins and fill levels"""
    
    def __init__(self, n_bins: int = 50, grid_size: int = 100, seed: int = 42):
        self.n_bins = n_bins
        self.grid_size = grid_size
        self.seed = seed
        np.random.seed(seed)
        
    def generate_bins(
        self,
        n_bins: int,
        x_range: Tuple[int, int] = (0, 100),
        y_range: Tuple[int, int] = (0, 100),
        fill_range: Tuple[int, int] = (10, 90),
        city_graph: Optional[nx.Graph] = None,
    ) -> pd.DataFrame:
        """Generate bins; if a city graph is provided, place bins on graph nodes."""
        if city_graph is not None and len(city_graph.nodes) > 0:
            nodes = list(city_graph.nodes())
            if n_bins <= len(nodes):
                chosen = random.sample(nodes, n_bins)
            else:
                chosen = nodes + random.choices(nodes, k=n_bins - len(nodes))
            xs = [p[0] for p in chosen]
            ys = [p[1] for p in chosen]
        else:
            xs = np.random.uniform(x_range[0], x_range[1], n_bins)
            ys = np.random.uniform(y_range[0], y_range[1], n_bins)

        fills = np.random.uniform(fill_range[0], fill_range[1], n_bins)
        return pd.DataFrame({"x": xs, "y": ys, "fill": fills})

    def add_uncertainty(self, fill_levels: np.ndarray, uncertainty_std: float = 15) -> np.ndarray:
        """Add uncertainty to fill level estimates."""
        noise = np.random.normal(0, uncertainty_std, len(fill_levels))
        noisy_fill = fill_levels + noise
        return np.clip(noisy_fill, 0, 100)
    
    def visualize_bins(self, bins_data: Dict[str, np.ndarray], 
                      title: str = "Waste Bin Distribution"):
        """Create scatter plot of bins with size proportional to fill level."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot bins
        scatter = ax.scatter(bins_data['x'], bins_data['y'], 
                           s=bins_data['fill']*2 + 20,  # Size proportional to fill
                           c=bins_data['fill'], 
                           cmap='RdYlGn_r',  # Red for full, green for empty
                           alpha=0.7,
                           edgecolors='black')
        
        # Add depot (center of grid)
        ax.scatter(self.grid_size//2, self.grid_size//2, 
                  s=300, marker='s', c='blue', label='Depot')
        
        # Labels and formatting
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Colorbar
        plt.colorbar(scatter, label='Fill Level (%)')
        
        plt.tight_layout()
        return fig, ax


class FillLevelPredictor:
    """Lightweight placeholder for ANN-style fill prediction (optional W2/W3)."""

    def __init__(self):
        self.weights = None

    def fit(self, features: np.ndarray, targets: np.ndarray):
        """Fit a linear model to mimic ANN training without extra deps."""
        # Add bias term
        X = np.hstack([features, np.ones((features.shape[0], 1))])
        # Least squares solution
        self.weights, *_ = np.linalg.lstsq(X, targets, rcond=None)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model not trained. Call fit() first.")
        X = np.hstack([features, np.ones((features.shape[0], 1))])
        return X @ self.weights

# ============================================================================
# EXPERT RULES MODULE (Week 2)
# ============================================================================

class ExpertRules:
    """Implement expert rules for waste collection decisions"""
    
    def __init__(self):
        self.rules = []
        self.initialize_rules()
    
    def initialize_rules(self):
        """Define expert rules for waste collection"""
        self.rules = [
            {
                'name': 'Skip Low Fill',
                'condition': lambda fill, capacity, distance: fill < 40,
                'action': 'skip',
                'priority': 1
            },
            {
                'name': 'Must Collect High Fill',
                'condition': lambda fill, capacity, distance: fill > 80,
                'action': 'collect',
                'priority': 3  # High priority
            },
            {
                'name': 'Skip If Far and Medium Fill',
                'condition': lambda fill, capacity, distance: 40 <= fill <= 60 and distance > 30,
                'action': 'skip',
                'priority': 2
            },
            {
                'name': 'Collect If Near Capacity End',
                'condition': lambda fill, capacity, distance: capacity < 20 and fill > 30,
                'action': 'skip',  # Save capacity for very full bins
                'priority': 2
            }
        ]
    
    def apply_rules(self, bin_fill: float, remaining_capacity: float, 
                   distance_to_bin: float) -> Tuple[str, str]:
        """Apply expert rules to decide collect/skip and return rule name."""
        applicable_rules = []
        
        for rule in self.rules:
            if rule['condition'](bin_fill, remaining_capacity, distance_to_bin):
                applicable_rules.append(rule)
        
        if not applicable_rules:
            return 'undecided', 'no_rule'
        
        # Sort by priority and get highest priority rule
        applicable_rules.sort(key=lambda x: x['priority'], reverse=True)
        top = applicable_rules[0]
        return top['action'], top['name']
    
    def resolve_conflicts(self, decisions: List[str]) -> str:
        """Resolve conflicts when multiple rules apply"""
        if 'collect' in decisions:
            return 'collect'
        elif 'skip' in decisions:
            return 'skip'
        return 'undecided'

# ============================================================================
# AGENT MODULE (Week 5)
# ============================================================================

class TruckAgent:
    """Agent representing a waste collection truck"""
    
    def __init__(self, agent_id: int, capacity: float, depot_location: Tuple[float, float]):
        self.agent_id = agent_id
        self.capacity = capacity
        self.current_load = 0
        self.current_location = depot_location
        self.route = [depot_location]
        self.collected_bins = []
        self.skipped_bins = []
        self.expert_rules = ExpertRules()
        self.total_distance = 0
        self.decision_log: List[Dict[str, Any]] = []
    
    def decide_action(self, bin_id: int, bin_fill: float, bin_location: Tuple[float, float], 
                     actual_fill: float = None) -> Tuple[str, str]:
        """Make decision to collect or skip a bin."""
        # Calculate distance to bin
        distance = np.sqrt((bin_location[0] - self.current_location[0])**2 +
                          (bin_location[1] - self.current_location[1])**2)
        
        # Apply expert rules
        decision, rule_name = self.expert_rules.apply_rules(bin_fill, 
                                                self.capacity - self.current_load,
                                                distance)
        
        # If undecided, use capacity-based heuristic
        if decision == 'undecided':
            if bin_fill > 50 and (self.current_load + actual_fill) <= self.capacity:
                decision = 'collect'
                rule_name = 'capacity_heuristic'
            else:
                decision = 'skip'
                rule_name = 'capacity_heuristic'
        
        return decision, rule_name
    
    def collect_bin(self, bin_id: int, actual_fill: float, bin_location: Tuple[float, float]):
        """Collect waste from a bin"""
        self.current_load += actual_fill
        self.collected_bins.append(bin_id)
        self.update_location(bin_location)
    
    def skip_bin(self, bin_id: int, bin_location: Tuple[float, float]):
        """Skip a bin"""
        self.skipped_bins.append(bin_id)
    
    def update_location(self, new_location: Tuple[float, float]):
        """Update truck location and track distance"""
        distance = np.sqrt((new_location[0] - self.current_location[0])**2 +
                          (new_location[1] - self.current_location[1])**2)
        self.total_distance += distance
        self.current_location = new_location
        self.route.append(new_location)
    
    def reset(self):
        """Reset agent for new simulation"""
        self.current_load = 0
        self.current_location = self.route[0]  # Reset to depot
        self.route = [self.current_location]
        self.collected_bins = []
        self.skipped_bins = []
        self.total_distance = 0
        self.decision_log = []

# ============================================================================
# GENETIC ALGORITHM MODULE (Week 4)
# ============================================================================

class GeneticAlgorithmOptimizer:
    """Genetic Algorithm for optimizing waste collection routes"""
    
    def __init__(self, population_size: int = 50, generations: int = 100,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.fitness_history = []
    
    class Chromosome:
        """Representation of a solution (route + decisions)"""
        def __init__(self, bin_count: int):
            self.bin_order = list(range(bin_count))
            random.shuffle(self.bin_order)
            self.collect_decisions = [random.random() > 0.3 for _ in range(bin_count)]
            self.fitness = float('inf')
        
        def copy(self):
            """Create a copy of the chromosome"""
            new_chrom = GeneticAlgorithmOptimizer.Chromosome(0)
            new_chrom.bin_order = self.bin_order.copy()
            new_chrom.collect_decisions = self.collect_decisions.copy()
            new_chrom.fitness = self.fitness
            return new_chrom
    
    def initialize_population(self, bin_count: int):
        """Initialize random population"""
        self.population = []
        for _ in range(self.population_size):
            chrom = self.Chromosome(bin_count)
            self.population.append(chrom)
    
    def evaluate_fitness(self, chromosome: Chromosome, bins_data: Dict[str, np.ndarray],
                        truck_capacity: float, uncertainty_std: float = 15) -> float:
        """Calculate fitness of a chromosome"""
        total_cost = 0
        penalty = 0
        
        # Simulate truck route
        truck = TruckAgent(1, truck_capacity, 
                          (np.mean(bins_data['x']), np.mean(bins_data['y'])))
        
        # Follow chromosome's order and decisions
        for i, bin_idx in enumerate(chromosome.bin_order):
            bin_id = bin_idx  # Assuming bin indices match IDs
            bin_loc = (bins_data['x'][bin_id], bins_data['y'][bin_id])
            estimated_fill = bins_data['fill'][bin_id]
            
            # Add uncertainty to actual fill
            actual_fill = estimated_fill + np.random.normal(0, uncertainty_std)
            actual_fill = np.clip(actual_fill, 0, 100)
            
            if chromosome.collect_decisions[i]:
                # Try to collect
                if truck.current_load + actual_fill <= truck.capacity:
                    truck.collect_bin(bin_id, actual_fill, bin_loc)
                else:
                    # Penalty for attempted collection when no capacity
                    penalty += 50
                    truck.skip_bin(bin_id, bin_loc)
            else:
                truck.skip_bin(bin_id, bin_loc)
                
                # Penalty if bin was actually full and we skipped
                if actual_fill > 90:
                    penalty += actual_fill  # Higher penalty for overflow risk
        
        # Return to depot
        truck.update_location(truck.route[0])
        
        # Calculate total cost: distance + penalties
        total_cost = truck.total_distance + penalty
        
        # Additional penalty for unserved full bins
        full_bins_skipped = 0
        for bin_id in truck.skipped_bins:
            actual_fill = bins_data['fill'][bin_id] + np.random.normal(0, uncertainty_std)
            actual_fill = np.clip(actual_fill, 0, 100)
            if actual_fill > 80:
                full_bins_skipped += 1
        
        total_cost += full_bins_skipped * 100  # Large penalty for skipping full bins
        
        return total_cost
    
    def select_parents(self) -> List[Chromosome]:
        """Select parents using tournament selection"""
        tournament_size = 3
        parents = []
        
        for _ in range(2):  # Select 2 parents
            tournament = random.sample(self.population, tournament_size)
            winner = min(tournament, key=lambda x: x.fitness)
            parents.append(winner.copy())
        
        return parents
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Perform ordered crossover for routes and uniform crossover for decisions"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Ordered crossover for bin order
        n = len(parent1.bin_order)
        start, end = sorted(random.sample(range(n), 2))
        
        child1_order = [-1] * n
        child2_order = [-1] * n
        
        # Copy segments
        child1_order[start:end] = parent1.bin_order[start:end]
        child2_order[start:end] = parent2.bin_order[start:end]
        
        # Fill remaining positions
        idx1, idx2 = end, end
        for i in range(n):
            pos = (end + i) % n
            if parent2.bin_order[pos] not in child1_order:
                child1_order[idx1 % n] = parent2.bin_order[pos]
                idx1 += 1
            if parent1.bin_order[pos] not in child2_order:
                child2_order[idx2 % n] = parent1.bin_order[pos]
                idx2 += 1
        
        # Uniform crossover for decisions
        child1_decisions = []
        child2_decisions = []
        for d1, d2 in zip(parent1.collect_decisions, parent2.collect_decisions):
            if random.random() > 0.5:
                child1_decisions.append(d1)
                child2_decisions.append(d2)
            else:
                child1_decisions.append(d2)
                child2_decisions.append(d1)
        
        # Create children
        child1 = self.Chromosome(0)
        child1.bin_order = child1_order
        child1.collect_decisions = child1_decisions
        
        child2 = self.Chromosome(0)
        child2.bin_order = child2_order
        child2.collect_decisions = child2_decisions
        
        return child1, child2
    
    def mutate(self, chromosome: Chromosome):
        """Apply mutation operators"""
        if random.random() < self.mutation_rate:
            # Swap mutation for route
            i, j = random.sample(range(len(chromosome.bin_order)), 2)
            chromosome.bin_order[i], chromosome.bin_order[j] = \
                chromosome.bin_order[j], chromosome.bin_order[i]
        
        if random.random() < self.mutation_rate:
            # Bit flip mutation for decisions
            idx = random.randint(0, len(chromosome.collect_decisions) - 1)
            chromosome.collect_decisions[idx] = not chromosome.collect_decisions[idx]
    
    def evolve(self, bins_data: Dict[str, np.ndarray], truck_capacity: float):
        """Run genetic algorithm evolution"""
        bin_count = len(bins_data['x'])
        self.initialize_population(bin_count)
        
        for generation in range(self.generations):
            # Evaluate fitness
            for chrom in self.population:
                chrom.fitness = self.evaluate_fitness(chrom, bins_data, truck_capacity)
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness)
            self.fitness_history.append(self.population[0].fitness)
            
            # Create next generation
            new_population = [self.population[0].copy()]  # Elitism
            
            while len(new_population) < self.population_size:
                # Selection
                parents = self.select_parents()
                
                # Crossover
                children = self.crossover(parents[0], parents[1])
                
                # Mutation
                for child in children:
                    self.mutate(child)
                    new_population.append(child)
            
            # Keep population size constant
            self.population = new_population[:self.population_size]
            
            # Print progress
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {self.fitness_history[-1]:.2f}")
        
        return self.population[0]  # Return best chromosome

# ============================================================================
# SIMULATION MODULE (Weeks 5-7)
# ============================================================================

class WasteCollectionSimulation:
    """Main simulation integrating all components"""
    
    def __init__(self, n_bins: int = 30, n_trucks: int = 2, truck_capacity: float = 300):
        self.n_bins = n_bins
        self.n_trucks = n_trucks
        self.truck_capacity = truck_capacity
        self.bin_generator = WasteBinGenerator(n_bins=n_bins)
        self.ga_optimizer = GeneticAlgorithmOptimizer(population_size=30, generations=50)
        self.bins_data = None
        self.trucks = []
        self.city_graph = None
        self.distance_matrix = None
        self.x_range = (0, 100)
        self.y_range = (0, 100)
        self.initialize_simulation()
    
    def initialize_simulation(self):
        """Initialize simulation with bins and trucks"""
        # Generate a realistic road network for bin placement
        self.city_graph = generate_realistic_city(
            num_districts=max(self.n_bins * 2, 40),
            city_radius=int(self.bin_generator.grid_size * 1.5)
        )

        # Generate bins aligned to the city graph
        self.bins_data = self.bin_generator.generate_bins(
            self.n_bins, x_range=self.x_range, y_range=self.y_range, city_graph=self.city_graph
        )

        # Precompute distances (bins and depot)
        self.distance_matrix = self._compute_distance_matrix(self.bins_data)
        
        # Create trucks (agents)
        depot = (np.mean(self.bins_data['x']), np.mean(self.bins_data['y']))
        for i in range(self.n_trucks):
            truck = TruckAgent(i + 1, self.truck_capacity, depot)
            self.trucks.append(truck)

    def _compute_distance_matrix(self, bins_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute Euclidean distance matrix for bins and depot."""
        coords = np.stack([bins_data['x'], bins_data['y']], axis=1)
        diff = coords[:, None, :] - coords[None, :, :]
        bin_to_bin = np.sqrt(np.sum(diff ** 2, axis=2))
        depot = np.array([np.mean(bins_data['x']), np.mean(bins_data['y'])])
        depot_to_bin = np.sqrt(np.sum((coords - depot) ** 2, axis=1))
        return {'bin_to_bin': bin_to_bin, 'depot_to_bin': depot_to_bin, 'depot': depot}

    def build_feature_matrix(self) -> np.ndarray:
        """Feature builder for optional ANN predictor (coords + current fill)."""
        x = self.bins_data['x'] / max(self.bin_generator.grid_size, 1)
        y = self.bins_data['y'] / max(self.bin_generator.grid_size, 1)
        fill = self.bins_data['fill'] / 100.0
        return np.stack([x, y, fill], axis=1)
    
    def run_agent_only_simulation(self, uncertainty_std: float = 15):
        """Run simulation using only agent rules"""
        results = []
        before_fill = self.bins_data['fill'].copy()
        after_fill = before_fill.copy()
        overflow_skipped = 0
        capacity_violations = 0
        
        for truck in self.trucks:
            truck.reset()
            
            # Create a random bin order
            bin_order = list(range(self.n_bins))
            random.shuffle(bin_order)
            
            for bin_id in bin_order:
                if bin_id in truck.collected_bins or bin_id in truck.skipped_bins:
                    continue
                
                bin_loc = (self.bins_data['x'][bin_id], self.bins_data['y'][bin_id])
                estimated_fill = self.bins_data['fill'][bin_id]
                
                # Add uncertainty
                actual_fill = estimated_fill + np.random.normal(0, uncertainty_std)
                actual_fill = np.clip(actual_fill, 0, 100)
                
                # Agent decision
                decision, rule_name = truck.decide_action(bin_id, estimated_fill, bin_loc, actual_fill)
                
                if decision == 'collect' and (truck.current_load + actual_fill) <= truck.capacity:
                    truck.collect_bin(bin_id, actual_fill, bin_loc)
                    after_fill[bin_id] = max(0, after_fill[bin_id] - actual_fill)
                else:
                    truck.skip_bin(bin_id, bin_loc)
                    if decision == 'collect':
                        capacity_violations += 1
                    if actual_fill > 90:
                        overflow_skipped += 1

                truck.decision_log.append({
                    'bin_id': bin_id,
                    'decision': decision,
                    'rule': rule_name,
                    'estimated_fill': estimated_fill,
                    'actual_fill': actual_fill,
                    'distance': np.sqrt((bin_loc[0] - truck.current_location[0])**2 + (bin_loc[1] - truck.current_location[1])**2),
                    'load_after': truck.current_load
                })
            
            # Return to depot
            truck.update_location(truck.route[0])
            
            results.append({
                'truck_id': truck.agent_id,
                'collected': len(truck.collected_bins),
                'skipped': len(truck.skipped_bins),
                'distance': truck.total_distance,
                'load': truck.current_load,
                'rule_trace': truck.decision_log,
                'before_fill': before_fill,
                'after_fill': after_fill,
                'overflow_skipped': overflow_skipped,
                'capacity_violations': capacity_violations
            })
        
        return results
    
    def run_ga_only_simulation(self, uncertainty_std: float = 15):
        """Run simulation using only GA optimization"""
        best_chromosome = self.ga_optimizer.evolve(self.bins_data, self.truck_capacity)
        
        # Simulate best chromosome
        truck = self.trucks[0]
        truck.reset()
        before_fill = self.bins_data['fill'].copy()
        after_fill = before_fill.copy()
        overflow_skipped = 0
        capacity_violations = 0
        
        for i, bin_idx in enumerate(best_chromosome.bin_order):
            bin_id = bin_idx
            bin_loc = (self.bins_data['x'][bin_id], self.bins_data['y'][bin_id])
            estimated_fill = self.bins_data['fill'][bin_id]
            
            actual_fill = estimated_fill + np.random.normal(0, uncertainty_std)
            actual_fill = np.clip(actual_fill, 0, 100)
            
            if best_chromosome.collect_decisions[i]:
                if truck.current_load + actual_fill <= truck.capacity:
                    truck.collect_bin(bin_id, actual_fill, bin_loc)
                    after_fill[bin_id] = max(0, after_fill[bin_id] - actual_fill)
                else:
                    truck.skip_bin(bin_id, bin_loc)
                    capacity_violations += 1
            else:
                truck.skip_bin(bin_id, bin_loc)
                if actual_fill > 90:
                    overflow_skipped += 1
        
        truck.update_location(truck.route[0])
        
        return {
            'method': 'GA Only',
            'fitness': best_chromosome.fitness,
            'collected': len(truck.collected_bins),
            'distance': truck.total_distance,
            'load': truck.current_load,
            'route': truck.route,
            'chromosome': best_chromosome,
            'rule_trace': [],
            'before_fill': before_fill,
            'after_fill': after_fill,
            'overflow_skipped': overflow_skipped,
            'capacity_violations': capacity_violations
        }
    
    def run_hybrid_simulation(self, uncertainty_std: float = 15):
        """Run hybrid simulation (GA + Agent rules)"""
        # First get GA optimized route
        ga_result = self.run_ga_only_simulation(uncertainty_std)
        chromosome = ga_result['chromosome']
        
        # Then let agents make final decisions based on rules
        truck = self.trucks[0]
        truck.reset()
        before_fill = self.bins_data['fill'].copy()
        after_fill = before_fill.copy()
        overflow_skipped = 0
        capacity_violations = 0
        
        for i, bin_idx in enumerate(chromosome.bin_order):
            bin_id = bin_idx
            bin_loc = (self.bins_data['x'][bin_id], self.bins_data['y'][bin_id])
            estimated_fill = self.bins_data['fill'][bin_id]
            
            actual_fill = estimated_fill + np.random.normal(0, uncertainty_std)
            actual_fill = np.clip(actual_fill, 0, 100)
            
            # Agent makes final decision
            decision, rule_name = truck.decide_action(bin_id, estimated_fill, bin_loc, actual_fill)
            
            if decision == 'collect' and (truck.current_load + actual_fill) <= truck.capacity:
                truck.collect_bin(bin_id, actual_fill, bin_loc)
                after_fill[bin_id] = max(0, after_fill[bin_id] - actual_fill)
            else:
                truck.skip_bin(bin_id, bin_loc)
                if decision == 'collect':
                    capacity_violations += 1
                if actual_fill > 90:
                    overflow_skipped += 1

            truck.decision_log.append({
                'bin_id': bin_id,
                'decision': decision,
                'rule': rule_name,
                'estimated_fill': estimated_fill,
                'actual_fill': actual_fill,
                'distance': np.sqrt((bin_loc[0] - truck.current_location[0])**2 + (bin_loc[1] - truck.current_location[1])**2),
                'load_after': truck.current_load
            })
        
        truck.update_location(truck.route[0])
        
        return {
            'method': 'Hybrid (GA + Agent)',
            'fitness': ga_result['fitness'],
            'collected': len(truck.collected_bins),
            'distance': truck.total_distance,
            'load': truck.current_load,
            'route': truck.route,
            'rule_trace': truck.decision_log,
            'before_fill': before_fill,
            'after_fill': after_fill,
            'overflow_skipped': overflow_skipped,
            'capacity_violations': capacity_violations
        }
    
    def visualize_results(self, results: Dict[str, Any], method_name: str):
        """Visualize routes and results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Bin fill levels and route
        ax1 = axes[0]
        
        # Plot bins
        scatter = ax1.scatter(self.bins_data['x'], self.bins_data['y'],
                            s=self.bins_data['fill']*2 + 20,
                            c=self.bins_data['fill'],
                            cmap='RdYlGn_r',
                            alpha=0.7,
                            edgecolors='black')
        
        # Plot route
        route_x = [point[0] for point in results['route']]
        route_y = [point[1] for point in results['route']]
        ax1.plot(route_x, route_y, 'b-', alpha=0.6, linewidth=2, label='Route')
        ax1.scatter(route_x[0], route_y[0], s=300, marker='s', c='blue', label='Depot')
        
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title(f'{method_name} - Collection Route')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Fill Level (%)')
        
        # Plot 2: Performance metrics
        ax2 = axes[1]
        
        metrics = ['Collected Bins', 'Distance', 'Truck Load']
        values = [results['collected'], results['distance'], results['load']]
        
        bars = ax2.bar(metrics, values, color=['green', 'orange', 'blue'])
        ax2.set_ylabel('Value')
        ax2.set_title(f'Performance Metrics\nFitness: {results["fitness"]:.2f}')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{method_name.replace(" ", "_")}_results.png', dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_before_after(self, before_fill: np.ndarray, after_fill: np.ndarray, method_name: str):
        """Plot bin fill levels before and after collection."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        cmap = 'RdYlGn_r'
        for idx, (fills, title) in enumerate(zip([before_fill, after_fill], ['Before', 'After'])):
            ax = axes[idx]
            scatter = ax.scatter(self.bins_data['x'], self.bins_data['y'],
                                 s=fills*2 + 20,
                                 c=fills,
                                 cmap=cmap,
                                 alpha=0.8,
                                 edgecolors='black')
            ax.set_title(f'{method_name} - {title} Collection')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Fill Level (%)')
        plt.tight_layout()
        plt.savefig(f'{method_name.replace(" ", "_")}_before_after.png', dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_rule_firings(self, rule_trace: List[Dict[str, Any]], method_name: str):
        """Bar plot of how often each rule fired."""
        if not rule_trace:
            print("No rule trace to visualize.")
            return
        counts = defaultdict(int)
        for event in rule_trace:
            counts[event['rule']] += 1
        labels = list(counts.keys())
        values = [counts[k] for k in labels]
        plt.figure(figsize=(8, 5))
        plt.bar(labels, values, color='steelblue')
        plt.ylabel('Fires')
        plt.title(f'{method_name} - Rule Firing Counts')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(f'{method_name.replace(" ", "_")}_rule_firing.png', dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_agent_load_trace(self, rule_trace: List[Dict[str, Any]], method_name: str):
        """Line plot of truck load over decisions to show interactions."""
        if not rule_trace:
            print("No interaction trace to visualize.")
            return
        loads = [event['load_after'] for event in rule_trace]
        decisions = [event['decision'][0].upper() for event in rule_trace]
        plt.figure(figsize=(10, 4))
        plt.plot(range(len(loads)), loads, marker='o')
        for idx, dec in enumerate(decisions):
            plt.text(idx, loads[idx] + 2, dec, ha='center', fontsize=8)
        plt.xlabel('Decision step')
        plt.ylabel('Truck load')
        plt.title(f'{method_name} - Agent Interaction Trace')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{method_name.replace(" ", "_")}_agent_trace.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def compare_methods(self, n_runs: int = 10):
        """Compare different methods over multiple runs"""
        methods = ['Agent Only', 'GA Only', 'Hybrid']
        results_summary = {method: {'fitness': [], 'distance': [], 'collected': []} 
                          for method in methods}
        
        print("\n" + "="*50)
        print("COMPARING METHODS OVER MULTIPLE RUNS")
        print("="*50)
        
        for run in range(n_runs):
            print(f"\nRun {run + 1}/{n_runs}")
            
            # Agent Only
            agent_results = self.run_agent_only_simulation()
            total_distance = sum([r['distance'] for r in agent_results])
            total_collected = sum([r['collected'] for r in agent_results])
            # Simplified fitness calculation for comparison
            fitness = total_distance + (self.n_bins - total_collected) * 10
            
            results_summary['Agent Only']['fitness'].append(fitness)
            results_summary['Agent Only']['distance'].append(total_distance)
            results_summary['Agent Only']['collected'].append(total_collected)
            
            # GA Only
            ga_result = self.run_ga_only_simulation()
            results_summary['GA Only']['fitness'].append(ga_result['fitness'])
            results_summary['GA Only']['distance'].append(ga_result['distance'])
            results_summary['GA Only']['collected'].append(ga_result['collected'])
            
            # Hybrid
            hybrid_result = self.run_hybrid_simulation()
            results_summary['Hybrid']['fitness'].append(hybrid_result['fitness'])
            results_summary['Hybrid']['distance'].append(hybrid_result['distance'])
            results_summary['Hybrid']['collected'].append(hybrid_result['collected'])
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['fitness', 'distance', 'collected']
        titles = ['Fitness (Lower is Better)', 'Total Distance', 'Bins Collected']
        colors = ['red', 'blue', 'green']
        
        for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            ax = axes[idx]
            
            data = []
            for method in methods:
                data.append(results_summary[method][metric])
            
            bp = ax.boxplot(data, labels=methods, patch_artist=True)
            
            # Color boxes
            for patch, color in zip(bp['boxes'], colors[:len(methods)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax.set_ylabel(title.split('(')[0].strip())
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('method_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS (Mean +- Std)")
        print("="*50)
        
        for method in methods:
            print(f"\n{method}:")
            print(f"  Fitness: {np.mean(results_summary[method]['fitness']):.2f} +- "
                  f"{np.std(results_summary[method]['fitness']):.2f}")
            print(f"  Distance: {np.mean(results_summary[method]['distance']):.2f} +- "
                  f"{np.std(results_summary[method]['distance']):.2f}")
            print(f"  Collected: {np.mean(results_summary[method]['collected']):.2f} +- "
                  f"{np.std(results_summary[method]['collected']):.2f}")
        
        return results_summary
    
    def visualize_city_map(self, filename: str = "city_map_with_bins.png"):
        """Plot road graph and bins colored by estimated fill."""
        if self.city_graph is None or self.bins_data is None:
            print("City graph or bins not initialized.")
            return

        plt.figure(figsize=(10, 10))

        pos = {node: node for node in self.city_graph.nodes()}
        nx.draw_networkx_edges(self.city_graph, pos, alpha=0.4, width=1.2, edge_color="gray")
        nx.draw_networkx_nodes(self.city_graph, pos, node_size=8, node_color="black", alpha=0.6)

        scatter = plt.scatter(
            self.bins_data["x"],
            self.bins_data["y"],
            s=self.bins_data["fill"] * 2 + 20,
            c=self.bins_data["fill"],
            cmap="RdYlGn_r",
            alpha=0.8,
            edgecolors="black",
            label="Bins (est. fill)",
        )

        depot = self.distance_matrix["depot"]
        plt.scatter(depot[0], depot[1], s=300, marker="s", c="blue", label="Depot")

        plt.title("Road Network with Bin Fill Levels")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.colorbar(scatter, label="Estimated Fill (%)")
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.show()

# ============================================================================
# MAIN EXECUTION AND DEMONSTRATION
# ============================================================================

def main():
    """Main function to run the complete system"""
    print("="*60)
    print("SMART WASTE COLLECTION ROUTING SYSTEM")
    print("Intro to AI - Agent-Based Evolutionary Approach")
    print("="*60)
    
    # Initialize simulation
    print("\n1. Initializing simulation...")
    simulation = WasteCollectionSimulation(n_bins=30, n_trucks=1, truck_capacity=300)
    print(f"   City graph: {simulation.city_graph.number_of_nodes()} intersections, "
          f"{simulation.city_graph.number_of_edges()} roads")
    print(f"   Distance matrix shape: {simulation.distance_matrix['bin_to_bin'].shape}")

    # Optional ANN-style predictor demonstration
    predictor = FillLevelPredictor()
    features = simulation.build_feature_matrix()
    synthetic_future_fill = np.clip(simulation.bins_data['fill'] + np.random.normal(0, 5, len(simulation.bins_data['fill'])), 0, 100)
    predictor.fit(features, synthetic_future_fill)
    predicted = predictor.predict(features)
    print(f"   ANN placeholder trained. Predicted mean fill: {np.mean(predicted):.2f}%")
    
    # Visualize initial bin distribution (Week 1)
    print("\n2. Generating and visualizing bin distribution...")
    fig, ax = simulation.bin_generator.visualize_bins(
        simulation.bins_data, 
        "Initial Waste Bin Distribution with Fill Levels"
    )
    plt.savefig('bin_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Run Agent-Only simulation (Week 5)
    print("\n3. Running Agent-Only simulation...")
    agent_results = simulation.run_agent_only_simulation()
    for result in agent_results:
        print(f"  Truck {result['truck_id']}: Collected {result['collected']} bins, "
              f"Distance: {result['distance']:.2f}")
    
    # Run GA-Only simulation (Week 4)
    print("\n4. Running GA-Only simulation...")
    ga_result = simulation.run_ga_only_simulation()
    simulation.visualize_results(ga_result, ga_result['method'])
    simulation.visualize_before_after(ga_result['before_fill'], ga_result['after_fill'], ga_result['method'])
    print(f"  Fitness: {ga_result['fitness']:.2f}, "
          f"Collected: {ga_result['collected']} bins")
    
    # Run Hybrid simulation (Week 6)
    print("\n5. Running Hybrid (GA + Agent) simulation...")
    hybrid_result = simulation.run_hybrid_simulation()
    simulation.visualize_results(hybrid_result, hybrid_result['method'])
    simulation.visualize_before_after(hybrid_result['before_fill'], hybrid_result['after_fill'], hybrid_result['method'])
    simulation.visualize_rule_firings(hybrid_result['rule_trace'], hybrid_result['method'])
    simulation.visualize_agent_load_trace(hybrid_result['rule_trace'], hybrid_result['method'])
    print(f"  Fitness: {hybrid_result['fitness']:.2f}, "
          f"Collected: {hybrid_result['collected']} bins")
    
    # Compare methods (Week 8)
    print("\n6. Comparing methods over multiple runs...")
    comparison_results = simulation.compare_methods(n_runs=5)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - bin_distribution.png: Initial bin visualization")
    print("  - GA_Only_results.png: GA optimization results")
    print("  - Hybrid_results.png: Hybrid approach results")
    print("  - method_comparison.png: Statistical comparison")
    
    return simulation, comparison_results

if __name__ == "__main__":
    # Run the complete system
    simulation, results = main()

def generate_realistic_city(num_districts: int = 60, city_radius: int = 150) -> nx.Graph:
    """Generate a road network graph using Voronoi-based districts."""
    centers = np.random.normal(0, city_radius / 2, (num_districts, 2))
    vor = Voronoi(centers)
    G = nx.Graph()

    for ridge in vor.ridge_vertices:
        if -1 not in ridge:
            p1, p2 = vor.vertices[ridge[0]], vor.vertices[ridge[1]]
            if np.linalg.norm(p1) < city_radius and np.linalg.norm(p2) < city_radius:
                dist = np.linalg.norm(p1 - p2)
                u, v = tuple(np.round(p1, 2)), tuple(np.round(p2, 2))
                G.add_edge(u, v, weight=dist)

    main_city = max(nx.connected_components(G), key=len)
    G = G.subgraph(main_city).copy()
    G.graph['city_radius'] = city_radius
    return G

