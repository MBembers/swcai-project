import numpy as np
from typing import List, Tuple
from .city import City
from .agents import Truck
from .expert_rules import ExpertRules, Action
from .config import CONFIG

class Simulation:
    def __init__(self, city: City, truck: Truck):
        self.city = city
        self.truck = truck
        self.expert = ExpertRules(min_fill_threshold=CONFIG['expert_rules']['min_fill_threshold'])

    def _get_observed_fill_level(self, bin_obj) -> float:
        """
        Get the observed fill level with uncertainty.
        The true fill level is perturbed by a normal distribution (0 to 2 sigmas).
        """
        # Calculate sigma as a percentage of bin capacity from config
        sigma_ratio = CONFIG['uncertainty']['fill_level_sigma_ratio']
        sigma = bin_obj.capacity * sigma_ratio
        
        # Sample from normal distribution: mean = true fill level, std = sigma
        # Clamp to [0, capacity] to avoid negative or over-capacity values
        observed = np.random.normal(bin_obj.fill_level, sigma)
        observed = max(0.0, min(bin_obj.capacity, observed))
        
        return observed

    def simulate_route(self, proposed_sequence: List[int]) -> Tuple[float, float, int]:
        current_pos = self.city.depot
        current_load = 0.0
        total_distance = 0.0
        bins_collected = 0
        penalty = 0.0
        
        # We must penalize if the truck returns to depot too often due to poor ordering
        trips = 1

        for bin_id in proposed_sequence:
            target_bin = self.city.bins[bin_id]
            
            # --- Capacity Check with Observed Fill Level ---
            observed_fill = self._get_observed_fill_level(target_bin)
            
            # Skip if bin is empty
            if observed_fill <= 0:
                continue
            
            space_available = self.truck.capacity - current_load
            
            if observed_fill <= space_available:
                # Full amount fits - take everything
                dist = self.city.distance(current_pos, target_bin.pos)
                total_distance += dist
                current_pos = target_bin.pos
                current_load += observed_fill
                bins_collected += 1
            else:
                # Doesn't fit fully
                if space_available > 0:
                    # Take partial amount to fill truck to capacity
                    dist = self.city.distance(current_pos, target_bin.pos)
                    total_distance += dist
                    current_pos = target_bin.pos
                    amount_to_take = space_available
                    current_load += amount_to_take
                    bins_collected += 1
                
                # Return to depot
                dist_to_depot = self.city.distance(current_pos, self.city.depot)
                total_distance += dist_to_depot
                
                # Reset
                current_pos = self.city.depot
                current_load = 0.0
                trips += 1

        # Final return to depot
        total_distance += self.city.distance(current_pos, self.city.depot)

        # Heuristic Penalty: We want fewer trips if possible, but mainly shorter distance
        # Heavy penalty for uncollected bins is handled in get_fitness
        
        return total_distance, penalty, bins_collected

    def get_fitness(self, genome: List[int]) -> float:
        dist, penalty, collected = self.simulate_route(genome)
        
        # Fitness = Distance + Penalty for missed bins
        # Note: 'genome' here only contains the bins we INTENDED to visit.
        # But we compare against ALL active bins to ensure we didn't miss any.
        
        return dist + penalty

    def get_trip_segments(self, route_ids: List[int]):
        """Re-runs the logic to return clean segments for plotting"""
        segments = []
        current_segment = [self.city.depot]
        current_load = 0.0
        
        for b_id in route_ids:
            target_bin = self.city.bins[b_id]
            
            # Use observed fill level with uncertainty
            observed_fill = self._get_observed_fill_level(target_bin)
            
            # Skip if bin is empty
            if observed_fill <= 0:
                continue
            
            space_available = self.truck.capacity - current_load
            
            if observed_fill <= space_available:
                # Full amount fits
                current_segment.append(target_bin.pos)
                current_load += observed_fill
            else:
                # Doesn't fit fully
                if space_available > 0:
                    # Take partial amount to fill truck to capacity
                    current_segment.append(target_bin.pos)
                    current_load += space_available
                
                # Return to depot
                current_segment.append(self.city.depot)
                segments.append(current_segment)
                current_segment = [self.city.depot]
                current_load = 0.0
            
        current_segment.append(self.city.depot)
        segments.append(current_segment)
        return segments
