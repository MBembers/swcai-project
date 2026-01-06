from typing import List, Tuple
from .city import City
from .agents import Truck

class Simulation:
    def __init__(self, city: City, truck: Truck):
        self.city = city
        self.truck = truck

    def get_trip_segments(self, route_ids: List[int]) -> List[List[Tuple[float, float]]]:
        segments = []
        current_segment = [self.city.depot]
        current_load = 0.0
        visited_count = 0

        for b_id in route_ids:
            target_bin = self.city.bins[b_id]
            
            # Skip if bin is mostly empty
            if target_bin.fill_level < 10: # get_from_config
                continue

            # If bin exceeds capacity, return to depot first
            if current_load + target_bin.fill_level > self.truck.capacity:
                if len(current_segment) > 1:
                    current_segment.append(self.city.depot)
                    segments.append(current_segment)
                current_segment = [self.city.depot]
                current_load = 0.0
            
            current_segment.append(target_bin.pos)
            current_load += target_bin.fill_level
            visited_count += 1

        if len(current_segment) > 1:
            current_segment.append(self.city.depot)
            segments.append(current_segment)
            
        return segments

    def run_route(self, route_ids: List[int]) -> float:
        segments = self.get_trip_segments(route_ids)
        total_dist = 0.0
        bins_visited = 0
        
        for seg in segments:
            bins_visited += (len(seg) - 2) # Subtract depot start and end
            for i in range(len(seg) - 1):
                total_dist += city.distance(seg[i], seg[i+1])
        
        # IMPORTANT: Penalty for bins NOT visited. 
        # This forces the AI to include as many bins as possible in the route.
        unvisited_penalty = (len(self.city.bins) - bins_visited) * 500 # get_from_config
        
        if total_dist == 0:
            return 10000.0 # get_from_config
            
        return total_dist + unvisited_penalty
