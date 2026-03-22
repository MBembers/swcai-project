import numpy as np
import random
import math
from typing import List, Callable, Tuple, Any
import concurrent.futures

class LShadeOptimizer:
    def __init__(self, config: dict, bin_ids: List[int], fitness_fn: Callable, pop_size: int = None, baseline_route: List[int] = None, parallel_workers: int = 1):
        self.config = config
        self.bin_ids = bin_ids
        self.dim = len(bin_ids)
        self.fitness_fn = fitness_fn
        self.parallel_workers = parallel_workers
        
        # Configuration setup, default to some reasonable L-SHADE values if not present
        l_config = self.config.get('lshade', {})
        self.max_pop_size = pop_size if pop_size is not None else l_config.get('max_pop_size', max(100, 10 * self.dim))
        self.min_pop_size = l_config.get('min_pop_size', 4)
        self.H = l_config.get('memory_size', 5)
        self.p_best_rate = l_config.get('p_best_rate', 0.1)
        self.arc_rate = l_config.get('arc_rate', 2.0)
        self.max_generations = self.config.get('evolution', {}).get('generations', 100)
        self.max_evals = l_config.get('max_evals', self.max_pop_size * self.max_generations)
        
        self.pop_size = self.max_pop_size
        self.evals = 0
        
        # Initialize population
        self.population = np.random.rand(self.pop_size, self.dim)
        
        # Put baseline route in if provided
        if baseline_route is not None:
            # Create a corresponding continuous vector that sorts to the baseline route
            # Baseline route contains absolute bin_ids, we need to rank them
            rank = [baseline_route.index(b_id) for b_id in self.bin_ids]
            # Map rank to [0, 1]
            vec = np.array(rank) / (self.dim + 1.0)
            self.population[0] = vec
            
        self.fitness = np.zeros(self.pop_size)
        self.eval_population(list(range(self.pop_size)))
        
        # Archive
        self.archive = []
        
        # Historical memory
        self.M_CR = np.ones(self.H) * 0.5
        self.M_F = np.ones(self.H) * 0.5
        self.k = 0 # memory index

    def vec_to_route(self, vec: np.ndarray) -> List[int]:
        """Convert continuous vector to permutation of bin_ids."""
        return [self.bin_ids[i] for i in np.argsort(vec)]

    def eval_population(self, indices: List[int]):
        """Evaluate a list of candidate indices. (Can be parallelized later)"""
        if self.parallel_workers > 1:
            routes = [self.vec_to_route(self.population[i]) for i in indices]
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
                results = list(executor.map(self.fitness_fn, routes))
            for idx, res in zip(indices, results):
                self.fitness[idx] = res
            self.evals += len(indices)
        else:
            for i in indices:
                route = self.vec_to_route(self.population[i])
                self.fitness[i] = self.fitness_fn(route)
                self.evals += 1

    def evaluate_trials(self, trials: np.ndarray) -> np.ndarray:
        """Evaluate a full batch of trial vectors. Useful for parallelization hook."""
        trial_fitness = np.zeros(len(trials))
        routes = [self.vec_to_route(trial) for trial in trials]
        
        if self.parallel_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
                results = list(executor.map(self.fitness_fn, routes))
            for i, res in enumerate(results):
                trial_fitness[i] = res
            self.evals += len(trials)
        else:
            for i in range(len(trials)):
                trial_fitness[i] = self.fitness_fn(routes[i])
                self.evals += 1
        return trial_fitness

    def evolve(self, max_generations: int = None, patience: int = None) -> Tuple[List[int], int]:
        """
        Run L-SHADE generations.
        """
        if max_generations is not None:
            self.max_generations = max_generations
        
        patience = patience or self.config.get('evolution', {}).get('patience', 100)
        
        best_fitness = np.min(self.fitness)
        best_vector = self.population[np.argmin(self.fitness)]
        
        gens_without_improvement = 0
        gen = 0
        
        progress_interval = self.config.get('evolution', {}).get('progress_interval', 10)

        while self.evals < self.max_evals and gen < self.max_generations:
            gen += 1
            
            # 1. Generate CR and F for each individual
            cr_i = np.zeros(self.pop_size)
            f_i = np.zeros(self.pop_size)
            
            for i in range(self.pop_size):
                r = np.random.randint(0, self.H)
                mu_cr = self.M_CR[r]
                mu_f = self.M_F[r]
                
                # Sample CR from normal distribution, clamp to [0, 1]
                if mu_cr < 0:
                    cr_i[i] = 0.0
                else:
                    cr_i[i] = np.random.normal(mu_cr, 0.1)
                cr_i[i] = np.clip(cr_i[i], 0.0, 1.0)
                
                # Sample F from Cauchy distribution
                while True:
                    f_val = mu_f + 0.1 * np.random.standard_cauchy()
                    if f_val > 0:
                        break
                f_i[i] = min(f_val, 1.0)
            
            # 2. Mutation (current-to-pbest/1)
            trials = np.zeros_like(self.population)
            
            # Sort for pbest
            sorted_idx = np.argsort(self.fitness)
            p_num = max(1, int(round(self.p_best_rate * self.pop_size)))
            
            for i in range(self.pop_size):
                pbest_idx = np.random.choice(sorted_idx[:p_num])
                
                # r1, r2 selection
                r1_idx = np.random.randint(0, self.pop_size)
                while r1_idx == i:
                    r1_idx = np.random.randint(0, self.pop_size)
                    
                arc_pop_size = self.pop_size + len(self.archive)
                r2_idx = np.random.randint(0, arc_pop_size)
                while r2_idx == i or r2_idx == r1_idx:
                    r2_idx = np.random.randint(0, arc_pop_size)
                    
                x_pbest = self.population[pbest_idx]
                x_r1 = self.population[r1_idx]
                if r2_idx < self.pop_size:
                    x_r2 = self.population[r2_idx]
                else:
                    x_r2 = self.archive[r2_idx - self.pop_size]
                    
                v = self.population[i] + f_i[i] * (x_pbest - self.population[i]) + f_i[i] * (x_r1 - x_r2)
                
                # Boundary reflection
                for j in range(self.dim):
                    if v[j] < 0:
                        v[j] = (self.population[i, j] + 0.0) / 2.0
                    elif v[j] > 1:
                        v[j] = (self.population[i, j] + 1.0) / 2.0
                        
                # 3. Crossover (Binomial)
                j_rand = np.random.randint(0, self.dim)
                u = np.copy(self.population[i])
                
                for j in range(self.dim):
                    if np.random.rand() <= cr_i[i] or j == j_rand:
                        u[j] = v[j]
                
                trials[i] = u
                
            # 4. Selection
            trial_fitness = self.evaluate_trials(trials)
            
            S_CR = []
            S_F = []
            delta_f = []
            
            for i in range(self.pop_size):
                if trial_fitness[i] < self.fitness[i]:
                    self.archive.append(self.population[i])
                    
                    S_CR.append(cr_i[i])
                    S_F.append(f_i[i])
                    delta_f.append(self.fitness[i] - trial_fitness[i])
                    
                    self.population[i] = trials[i]
                    self.fitness[i] = trial_fitness[i]
                elif trial_fitness[i] == self.fitness[i]:
                    # To avoid stagnation
                    self.population[i] = trials[i]
            
            # Manage archive size
            max_arc_size = int(round(self.arc_rate * self.pop_size))
            while len(self.archive) > max_arc_size:
                pop_idx = np.random.randint(0, len(self.archive))
                self.archive.pop(pop_idx)
                
            # 5. Update Memory
            if len(S_CR) > 0:
                weights = np.array(delta_f) / np.sum(delta_f)
                
                # Weighted Lehmer mean for M_F
                m_f_new = np.sum(weights * (np.array(S_F)**2)) / np.sum(weights * np.array(S_F))
                # Weighted arithmetic mean for M_CR
                m_cr_new = np.sum(weights * np.array(S_CR))
                
                self.M_F[self.k] = m_f_new
                self.M_CR[self.k] = m_cr_new
                
                self.k = (self.k + 1) % self.H
                
            # 6. Linear Population Size Reduction (LPSR)
            new_pop_size = int(round(self.max_pop_size - (self.max_pop_size - self.min_pop_size) * (self.evals / self.max_evals)))
            if new_pop_size < self.min_pop_size:
                new_pop_size = self.min_pop_size
                
            if new_pop_size < self.pop_size:
                # Keep the best individuals
                sorted_idx = np.argsort(self.fitness)
                keep_idx = sorted_idx[:new_pop_size]
                self.population = self.population[keep_idx]
                self.fitness = self.fitness[keep_idx]
                self.pop_size = new_pop_size
                
                # Resize archive
                max_arc_size = int(round(self.arc_rate * self.pop_size))
                while len(self.archive) > max_arc_size:
                    self.archive.pop(np.random.randint(0, len(self.archive)))

            current_best_idx = np.argmin(self.fitness)
            current_best_fitness = self.fitness[current_best_idx]
            
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_vector = self.population[current_best_idx]
                gens_without_improvement = 0
            else:
                gens_without_improvement += 1
                
            if gen % progress_interval == 0:
                print(f"\r[LSHADE] Gen {gen:03} | Cost: {current_best_fitness:.2f} | Pop {self.pop_size} | No imrp: {gens_without_improvement}/{patience}        ", end="", flush=True)

            if gens_without_improvement >= patience:
                #print(f"\n[LSHADE] Stopping at generation {gen}: {patience} generations without improvement")
                break

        print("")
        return self.vec_to_route(best_vector), gen

class ALShadeOptimizer(LShadeOptimizer):
    def __init__(self, config: dict, bin_ids: List[int], fitness_fn: Callable, pop_size: int = None, baseline_route: List[int] = None, parallel_workers: int = 1):
        super().__init__(config, bin_ids, fitness_fn, pop_size, baseline_route, parallel_workers)
        # Probabilities for choosing mutation strategies
        self.p_mut1 = 0.5 # current-to-pbest/1
        self.p_mut2 = 0.5 # current-to-Amean/1
        
    def evolve(self, max_generations: int = None, patience: int = None) -> Tuple[List[int], int]:
        if max_generations is not None:
            self.max_generations = max_generations
        
        patience = patience or self.config.get('evolution', {}).get('patience', 100)
        
        best_fitness = np.min(self.fitness)
        best_vector = self.population[np.argmin(self.fitness)]
        
        gens_without_improvement = 0
        gen = 0
        
        progress_interval = self.config.get('evolution', {}).get('progress_interval', 10)

        while self.evals < self.max_evals and gen < self.max_generations:
            gen += 1
            
            cr_i = np.zeros(self.pop_size)
            f_i = np.zeros(self.pop_size)
            
            for i in range(self.pop_size):
                r = np.random.randint(0, self.H)
                mu_cr = self.M_CR[r]
                mu_f = self.M_F[r]
                
                if mu_cr < 0:
                    cr_i[i] = 0.0
                else:
                    cr_i[i] = np.random.normal(mu_cr, 0.1)
                cr_i[i] = np.clip(cr_i[i], 0.0, 1.0)
                
                while True:
                    f_val = mu_f + 0.1 * np.random.standard_cauchy()
                    if f_val > 0:
                        break
                f_i[i] = min(f_val, 1.0)
            
            trials = np.zeros_like(self.population)
            sorted_idx = np.argsort(self.fitness)
            p_num = max(1, int(round(self.p_best_rate * self.pop_size)))
            
            # calculate Amean for current-to-Amean/1
            # Amean is typically the mean of the top 50% individuals in the current population and archive
            all_vectors = np.vstack((self.population, self.archive)) if len(self.archive) > 0 else self.population
            x_amean = np.mean(all_vectors, axis=0)

            # Strategy selection
            mut_strategy = np.random.choice([1, 2], size=self.pop_size, p=[self.p_mut1, self.p_mut2])
            
            for i in range(self.pop_size):
                pbest_idx = np.random.choice(sorted_idx[:p_num])
                
                r1_idx = np.random.randint(0, self.pop_size)
                while r1_idx == i:
                    r1_idx = np.random.randint(0, self.pop_size)
                    
                arc_pop_size = self.pop_size + len(self.archive)
                r2_idx = np.random.randint(0, arc_pop_size)
                while r2_idx == i or r2_idx == r1_idx:
                    r2_idx = np.random.randint(0, arc_pop_size)
                    
                x_pbest = self.population[pbest_idx]
                x_r1 = self.population[r1_idx]
                if r2_idx < self.pop_size:
                    x_r2 = self.population[r2_idx]
                else:
                    x_r2 = self.archive[r2_idx - self.pop_size]
                    
                if mut_strategy[i] == 1:
                    # current-to-pbest/1
                    v = self.population[i] + f_i[i] * (x_pbest - self.population[i]) + f_i[i] * (x_r1 - x_r2)
                else:
                    # current-to-Amean/1
                    v = self.population[i] + f_i[i] * (x_amean - self.population[i]) + f_i[i] * (x_r1 - x_r2)
                
                # Boundary reflection
                for j in range(self.dim):
                    if v[j] < 0:
                        v[j] = (self.population[i, j] + 0.0) / 2.0
                    elif v[j] > 1:
                        v[j] = (self.population[i, j] + 1.0) / 2.0
                        
                j_rand = np.random.randint(0, self.dim)
                u = np.copy(self.population[i])
                
                for j in range(self.dim):
                    if np.random.rand() <= cr_i[i] or j == j_rand:
                        u[j] = v[j]
                
                trials[i] = u
                
            trial_fitness = self.evaluate_trials(trials)
            
            S_CR = []
            S_F = []
            delta_f = []
            succ_mut1 = 0
            succ_mut2 = 0
            sum_df_mut1 = 0
            sum_df_mut2 = 0
            
            for i in range(self.pop_size):
                if trial_fitness[i] < self.fitness[i]:
                    self.archive.append(self.population[i])
                    
                    df = self.fitness[i] - trial_fitness[i]
                    S_CR.append(cr_i[i])
                    S_F.append(f_i[i])
                    delta_f.append(df)
                    
                    if mut_strategy[i] == 1:
                        succ_mut1 += 1
                        sum_df_mut1 += df
                    else:
                        succ_mut2 += 1
                        sum_df_mut2 += df
                    
                    self.population[i] = trials[i]
                    self.fitness[i] = trial_fitness[i]
                elif trial_fitness[i] == self.fitness[i]:
                    self.population[i] = trials[i]
            
            # Adapt mutation strategy probabilities
            if sum_df_mut1 + sum_df_mut2 > 0:
                p1_update = sum_df_mut1 / (sum_df_mut1 + sum_df_mut2)
                p2_update = sum_df_mut2 / (sum_df_mut1 + sum_df_mut2)
                
                # smooth update
                c = 0.1
                self.p_mut1 = (1 - c) * self.p_mut1 + c * p1_update
                self.p_mut2 = (1 - c) * self.p_mut2 + c * p2_update
                
                # Ensure a minimal probability so neither strategy is completely abandoned
                self.p_mut1 = np.clip(self.p_mut1, 0.1, 0.9)
                self.p_mut2 = 1.0 - self.p_mut1
            
            max_arc_size = int(round(self.arc_rate * self.pop_size))
            while len(self.archive) > max_arc_size:
                pop_idx = np.random.randint(0, len(self.archive))
                self.archive.pop(pop_idx)
                
            if len(S_CR) > 0:
                weights = np.array(delta_f) / np.sum(delta_f)
                m_f_new = np.sum(weights * (np.array(S_F)**2)) / np.sum(weights * np.array(S_F))
                m_cr_new = np.sum(weights * np.array(S_CR))
                
                self.M_F[self.k] = m_f_new
                self.M_CR[self.k] = m_cr_new
                self.k = (self.k + 1) % self.H
                
            new_pop_size = int(round(self.max_pop_size - (self.max_pop_size - self.min_pop_size) * (self.evals / self.max_evals)))
            if new_pop_size < self.min_pop_size:
                new_pop_size = self.min_pop_size
                
            if new_pop_size < self.pop_size:
                sorted_idx = np.argsort(self.fitness)
                keep_idx = sorted_idx[:new_pop_size]
                self.population = self.population[keep_idx]
                self.fitness = self.fitness[keep_idx]
                self.pop_size = new_pop_size
                
                max_arc_size = int(round(self.arc_rate * self.pop_size))
                while len(self.archive) > max_arc_size:
                    self.archive.pop(np.random.randint(0, len(self.archive)))

            current_best_idx = np.argmin(self.fitness)
            current_best_fitness = self.fitness[current_best_idx]
            
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_vector = self.population[current_best_idx]
                gens_without_improvement = 0
            else:
                gens_without_improvement += 1
                
            if gen % progress_interval == 0:
                print(f"\r[ALSHADE] Gen {gen:03} | Cost: {current_best_fitness:.2f} | Pop {self.pop_size} | No imrp: {gens_without_improvement}/{patience}        ", end="", flush=True)

            if gens_without_improvement >= patience:
                break

        print("")
        return self.vec_to_route(best_vector), gen
