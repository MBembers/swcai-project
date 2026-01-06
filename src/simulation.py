from typing import List, Tuple
from .city import City
from .agents import Truck

class Simulation:
    def __init__(self, city: City, truck: Truck):
        self.city = city
        self.truck = truck

    def get_trip_segments(self, route_ids: List[int]) -> List[List[Tuple[float, float]]]:
        segments = []
        # Start at depot
        current_segment = [self.city.depot]
        current_load = 0.0
        
        for b_id in route_ids:
            target_bin = self.city.bins[b_id]
            
            # Skip if bin is mostly empty (Evolution optimization logic)
            if target_bin.fill_level < 10: 
                continue

            # If adding this bin exceeds capacity, return to depot first
            if current_load + target_bin.fill_level > self.truck.capacity:
                # Close current trip
                current_segment.append(self.city.depot)
                segments.append(current_segment)
                
                # Start new trip
                current_segment = [self.city.depot, target_bin.pos]
                current_load = target_bin.fill_level
            else:
                # Add to current trip
                current_segment.append(target_bin.pos)
                current_load += target_bin.fill_level

        # Finish final segment
        if len(current_segment) > 1:
            current_segment.append(self.city.depot)
            segments.append(current_segment)
            
        return segments

    def run_route(self, route_ids: List[int]) -> float:
        segments = self.get_trip_segments(route_ids)
        total_dist = 0.0
        bins_visited = 0
        
        for seg in segments:
            # Each segment is Depot -> Bins... -> Depot
            # Number of bins visited in this segment is len(seg) - 2 (start/end depot)
            if len(seg) > 2:
                bins_visited += (len(seg) - 2)
                for i in range(len(seg) - 1):
                    # FIX: use self.city.distance
                    total_dist += self.city.distance(seg[i], seg[i+1])
        
        # Penalty for bins NOT visited
        unvisited_penalty = (len(self.city.bins) - bins_visited) * 200 
        
        if total_dist == 0 and bins_visited == 0:
            return 20000.0 
            
        return total_dist + unvisited_penalty
