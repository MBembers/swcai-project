import random
from typing import List, Callable

class GeneticOptimizer:
    def __init__(self, bin_ids: List[int], fitness_fn: Callable, pop_size: int = 150):
        self.bin_ids = bin_ids
        self.fitness_fn = fitness_fn
        self.pop_size = pop_size
        self.population = [self._random_genome() for _ in range(pop_size)]

    def _random_genome(self) -> List[int]:
        genome = self.bin_ids.copy()
        random.shuffle(genome)
        return genome

    def crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        # Ordered Crossover (OX) is essential for permutation problems like routing
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        
        child = [None] * size
        child[a:b] = p1[a:b]
        
        p2_idx = 0
        for i in range(size):
            if child[i] is None:
                while p2[p2_idx] in child:
                    p2_idx += 1
                child[i] = p2[p2_idx]
        return child

    def mutate(self, genome: List[int]):
        # Swap Mutation: Swaps two random bins in the sequence
        if random.random() < 0.2:
            i, j = random.sample(range(len(genome)), 2)
            genome[i], genome[j] = genome[j], genome[i]
        
        # Inversion Mutation: Reverses a sub-section of the route
        if random.random() < 0.1:
            i, j = sorted(random.sample(range(len(genome)), 2))
            genome[i:j] = reversed(genome[i:j])

    def evolve(self, generations: int = 200):
        for gen in range(generations):
            # Sort by fitness (lower is better)
            self.population.sort(key=self.fitness_fn)
            
            # Elitism: Keep the top 5
            next_gen = self.population[:5]
            
            while len(next_gen) < self.pop_size:
                # Tournament Selection
                parent1 = min(random.sample(self.population, 5), key=self.fitness_fn)
                parent2 = min(random.sample(self.population, 5), key=self.fitness_fn)
                
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                next_gen.append(child)
            
            self.population = next_gen
            if gen % 20 == 0:
                print(f"Gen {gen} | Best Distance: {self.fitness_fn(self.population[0]):.2f}")
                
        return self.population[0]
