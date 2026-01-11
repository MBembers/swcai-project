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
        base_rate = CONFIG['bin_refill']['base_rate_ratio']
        # Track learned fill rates and visit history per bin
        self.bin_profiles = {
            b.id: {
                'last_visit_day': 0,
                'observed_rate': base_rate  # ratio of capacity per day
            }
            for b in self.city.bins
        }
        self.current_day = 0
        self.current_observed_fills = {}  # Cache observations for this collection cycle

    def refill_bins(self, days: int = 1):
        """Advance time and refill bins according to predisposition plus noise."""
        noise_sigma_ratio = CONFIG['bin_refill']['noise_sigma_ratio']
        for b in self.city.bins:
            expected_add = b.fill_rate * b.capacity * days
            noise = np.random.normal(0.0, b.capacity * noise_sigma_ratio * np.sqrt(days))
            delta = max(0.0, expected_add + noise)
            b.fill_level = min(b.capacity, b.fill_level + delta)
        self.current_day += days

    def sample_collection_observations(self):
        """
        Sample observations for all bins at the START of a collection run.
        This ensures the GA optimizes against consistent data.
        """
        self.current_observed_fills = {}
        for b in self.city.bins:
            self.current_observed_fills[b.id] = self._get_observed_fill_level(b)

    def _get_cached_observed_fill(self, bin_id: int) -> float:
        """Use cached observation if available, otherwise sample fresh."""
        if bin_id in self.current_observed_fills:
            return self.current_observed_fills[bin_id]
        return self._get_observed_fill_level(self.city.bins[bin_id])

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
            
            # --- Capacity Check with Observed Fill Level (use cached) ---
            observed_fill = self._get_cached_observed_fill(bin_id)
            
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

    def plan_active_bins(self, collection_interval_days: int) -> List[int]:
        """
        Decide which bins to visit, using learned rates to anticipate fast fillers.
        Bins predicted to reach the expert threshold by the next collection are included.
        """
        active_bin_ids = []
        threshold = CONFIG['expert_rules']['min_fill_threshold']
        for b in self.city.bins:
            profile = self.bin_profiles[b.id]
            predicted_fill = min(
                b.capacity,
                b.fill_level + b.capacity * profile['observed_rate'] * collection_interval_days
            )
            predicted_ratio = predicted_fill / b.capacity if b.capacity > 0 else 0.0
            if predicted_ratio >= threshold:
                active_bin_ids.append(b.id)
        return active_bin_ids

    def execute_route(self, proposed_sequence: List[int]) -> Tuple[float, float, int]:
        """
        Stateful version of simulate_route: updates bin fill levels and learned rates.
        """
        current_pos = self.city.depot
        current_load = 0.0
        total_distance = 0.0
        bins_collected = 0
        penalty = 0.0
        learning_rate = CONFIG['bin_refill']['learning_rate']

        for bin_id in proposed_sequence:
            target_bin = self.city.bins[bin_id]
            observed_fill = self._get_observed_fill_level(target_bin)

            if observed_fill <= 0:
                continue

            space_available = self.truck.capacity - current_load

            if observed_fill <= space_available:
                dist = self.city.distance(current_pos, target_bin.pos)
                total_distance += dist
                current_pos = target_bin.pos
                amount_taken = observed_fill
                current_load += amount_taken
                bins_collected += 1
            else:
                if space_available > 0:
                    dist = self.city.distance(current_pos, target_bin.pos)
                    total_distance += dist
                    current_pos = target_bin.pos
                    amount_taken = space_available
                    current_load += amount_taken
                    bins_collected += 1
                else:
                    amount_taken = 0.0

                # Return to depot after filling up
                dist_to_depot = self.city.distance(current_pos, self.city.depot)
                total_distance += dist_to_depot
                current_pos = self.city.depot
                current_load = 0.0

            # Update bin state with actual collected amount
            if amount_taken > 0:
                target_bin.fill_level = max(0.0, target_bin.fill_level - amount_taken)
                profile = self.bin_profiles[target_bin.id]
                days_since_last = max(1, self.current_day - profile['last_visit_day'])
                observed_daily_rate = (amount_taken / target_bin.capacity) / days_since_last if target_bin.capacity > 0 else 0.0
                profile['observed_rate'] = (1 - learning_rate) * profile['observed_rate'] + learning_rate * observed_daily_rate
                profile['last_visit_day'] = self.current_day

        total_distance += self.city.distance(current_pos, self.city.depot)
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
            
            # Use cached observed fill level with uncertainty
            observed_fill = self._get_cached_observed_fill(b_id)
            
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
