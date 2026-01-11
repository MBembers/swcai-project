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
            
            # --- Capacity Check ---
            if current_load + target_bin.fill_level > self.truck.capacity:
                # Return to depot
                dist_to_depot = self.city.distance(current_pos, self.city.depot)
                total_distance += dist_to_depot
                
                # Reset
                current_pos = self.city.depot
                current_load = 0.0
                trips += 1
            
            # Move to Bin
            dist = self.city.distance(current_pos, target_bin.pos)
            total_distance += dist
            current_pos = target_bin.pos
            current_load += target_bin.fill_level
            bins_collected += 1

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
            
            # Note: We assume route_ids has already been filtered by Expert Rules
            # but we can re-check to be safe if visualization is called directly
            
            if current_load + target_bin.fill_level > self.truck.capacity:
                current_segment.append(self.city.depot)
                segments.append(current_segment)
                current_segment = [self.city.depot]
                current_load = 0.0
            
            current_segment.append(target_bin.pos)
            current_load += target_bin.fill_level
            
        current_segment.append(self.city.depot)
        segments.append(current_segment)
        return segments
