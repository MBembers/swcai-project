import numpy as np
from typing import List
from .city import CityGrid
from .utils import euclidean_distance

def get_greedy_route(city: CityGrid, truck_capacity: float) -> List[int]:
    """Greedy baseline: always go to the fullest bin next."""
    bins_to_visit = list(city.bins)
    route = []
    current_pos = city.depot
    current_load = 0.0
    
    while bins_to_visit:
        # Sort by fill level (descending)
        bins_to_visit.sort(key=lambda b: b.get_noisy_fill(0), reverse=True)
        next_bin = bins_to_visit.pop(0)
        
        if current_load + next_bin.get_noisy_fill(0) > truck_capacity:
            break
            
        route.append(next_bin.id)
        current_load += next_bin.get_noisy_fill(0)
        
    return route
