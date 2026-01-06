from typing import List, Tuple
from .city import Bin

class Truck:
    def __init__(self, truck_id: int, start_pos: Tuple[float, float], capacity: float = 500.0):
        self.id = truck_id
        self.start_pos = start_pos
        self.pos = start_pos
        self.capacity = capacity
        self.current_load = 0.0
        self.route: List[int] = []  # List of Bin IDs
        self.distance_traveled = 0.0


