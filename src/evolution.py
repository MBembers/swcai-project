import random
from typing import List, Callable
from .config import CONFIG

class GeneticOptimizer:
    def __init__(self, bin_ids: List[int], fitness_fn: Callable, pop_size: int = None):
        if pop_size is None:
            pop_size = CONFIG['evolution']['pop_size']
        self.bin_ids = bin_ids
        self.fitness_fn = fitness_fn
        self.pop_size = pop_size # Reduced from 150 to 50 for speed
        self.population = [self._random_genome() for _ in range(pop_size)]

    def _random_genome(self) -> List[int]:
        genome = self.bin_ids.copy()
        random.shuffle(genome)
        return genome

    def evolve(self, generations: int = None):
        if generations is None:
            generations = CONFIG['evolution']['generations']
        # Calculate fitness
        for gen in range(generations):
            # Calculate fitness
            scores = [(genome, self.fitness_fn(genome)) for genome in self.population]
            scores.sort(key=lambda x: x[1]) # Lower is better
            
            # Elitism: Keep top N
            next_gen = [s[0] for s in scores[:CONFIG['evolution']['elitism_count']]]
            
            # Breeding
            while len(next_gen) < self.pop_size:
                parent1 = random.choice(scores[:CONFIG['evolution']['tournament_selection_size']])[0]
                parent2 = random.choice(scores[:CONFIG['evolution']['tournament_selection_size']])[0]
                
                child = self._crossover(parent1, parent2)
                self._mutate(child)
                next_gen.append(child)
            
            self.population = next_gen
            if gen % CONFIG['evolution']['progress_interval'] == 0:
                print(f"Gen {gen} | Cost: {scores[0][1]:.2f}")
                
        return scores[0][0]

    def _crossover(self, p1, p2):
        # Order Crossover (OX)
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

    def _mutate(self, genome):
        if random.random() < CONFIG['evolution']['mutation_probability']:
            i, j = random.sample(range(len(genome)), 2)
            genome[i], genome[j] = genome[j], genome[i]
