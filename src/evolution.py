import random
from typing import List, Callable, Tuple

class GeneticOptimizer:
    def __init__(self,config:dict, bin_ids: List[int], fitness_fn: Callable, pop_size: int = None, baseline_route: List[int] = None):
        self.config = config
        if pop_size is None:
            pop_size = self.config['evolution']['pop_size']
        self.bin_ids = bin_ids
        self.fitness_fn = fitness_fn
        self.pop_size = pop_size
        
        self.population = [self._random_genome() for _ in range(pop_size)]
        if baseline_route:
            self.population[0] = baseline_route.copy()

    def _random_genome(self) -> List[int]:
        genome = self.bin_ids.copy()
        random.shuffle(genome)
        return genome

    def evolve(self, max_generations: int = None, patience: int = None) -> Tuple[List[int], int]:
        """
        Evolve population until stopping criterion is met.
        
        Args:
            max_generations: Maximum generations to run (safety limit)
            patience: Stop if no improvement for this many generations
            
        Returns:
            (best_route, generations_run)
        """
        if max_generations is None:
            max_generations = self.config['evolution']['generations']
        if patience is None:
            patience = self.config['evolution'].get('patience', 100)
        
        best_fitness = float('inf')
        generations_without_improvement = 0
        best_genome = None
        
        for gen in range(max_generations):
            # Calculate fitness
            scores = [(genome, self.fitness_fn(genome)) for genome in self.population]
            scores.sort(key=lambda x: x[1])  # Lower is better
            
            current_best_fitness = scores[0][1]
            
            # Check for improvement
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                generations_without_improvement = 0
                best_genome = scores[0][0]
            else:
                generations_without_improvement += 1
            
            # Elitism: Keep top N
            next_gen = [s[0] for s in scores[:self.config['evolution']['elitism_count']]]
            
            # Breeding
            while len(next_gen) < self.pop_size:
                parent1 = random.choice(scores[:self.config['evolution']['tournament_selection_size']])[0]
                parent2 = random.choice(scores[:self.config['evolution']['tournament_selection_size']])[0]
                
                child = self._crossover(parent1, parent2)
                self._mutate(child)
                
                # Apply two-opt local search based on configured rate
                two_opt_rate = self.config['evolution'].get('two_opt_rate', 0)
                if random.random() < (two_opt_rate / 100.0):
                    self._two_opt(child)
                next_gen.append(child)
            
            self.population = next_gen
            
            if gen % self.config['evolution']['progress_interval'] == 0:
                #print(f"Gen {gen} | Cost: {current_best_fitness:.2f} | No improvement: {generations_without_improvement}/{patience}")
                print(f"\rGen {gen:03} | Cost: {current_best_fitness:.2f} | No improvement: {generations_without_improvement}/{patience}        ", end="", flush=True)
            
            # Early stopping criterion
            if generations_without_improvement >= patience:
                print(f"Stopping at generation {gen}: {patience} generations without improvement")
                break
        
        return best_genome, gen + 1

    def _crossover(self, p1, p2):
        crossover_type = self.config['evolution'].get('crossover_type', 'order')
        
        if crossover_type == '2-point':
            return self._two_point_crossover(p1, p2)
        else:
            return self._order_crossover(p1, p2)
    
    def _order_crossover(self, p1, p2):
        """Order Crossover (OX) - preserves relative order"""
        start, end = sorted(random.sample(range(len(p1)), 2))
        child = [None]*len(p1)
        child[start:end] = p1[start:end]
        
        pointer = 0
        for gene in p2:
            if gene not in child[start:end]:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = gene
        return child
    
    def _two_point_crossover(self, p1, p2):
        """Two-point crossover for permutation encoding"""
        n = len(p1)
        # Select two crossover points
        point1, point2 = sorted(random.sample(range(n), 2))
        
        # Create child with middle segment from p1
        child = [None] * n
        child[point1:point2] = p1[point1:point2]
        
        # Fill remaining positions with genes from p2 in order
        p2_genes = [g for g in p2 if g not in child[point1:point2]]
        
        # Fill before first point
        child[:point1] = p2_genes[:point1]
        # Fill after second point
        child[point2:] = p2_genes[point1:]
        
        return child

    def _mutate(self, genome):
        if random.random() < self.config['evolution']['mutation_probability']:
            i, j = random.sample(range(len(genome)), 2)
            genome[i], genome[j] = genome[j], genome[i]
    
    def _two_opt(self, genome):
        """Apply two-opt local search improvement"""
        n = len(genome)
        improved = True
        
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    # Reverse segment between i and j
                    new_genome = genome[:i+1] + genome[i+1:j+1][::-1] + genome[j+1:]
                    
                    # Check if this improves fitness
                    if self.fitness_fn(new_genome) < self.fitness_fn(genome):
                        genome[:] = new_genome
                        improved = True
                        break
                if improved:
                    break
        
        return genome
