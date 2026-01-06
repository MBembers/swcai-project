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

    def move_to(self, target_pos: Tuple[float, float]):
        from .utils import euclidean_distance
        self.distance_traveled += euclidean_distance(self.pos, target_pos)
        self.pos = target_pos

    def collect(self, waste_bin: Bin):
        amount = min(waste_bin.fill_level, self.capacity - self.current_load)
        waste_bin.fill_level -= amount
        self.current_load += amount

    def is_full(self) -> bool:
        return self.current_load >= self.capacity

    def reset(self):
        self.pos = self.start_pos
        self.current_load = 0.0
        self.distance_traveled = 0.0
