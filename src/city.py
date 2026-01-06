import numpy as np
from scipy.spatial import Voronoi
import random
import networkx as nx
from enum import Enum
from typing import Tuple, List, Optional

class CityType(Enum):
    REALISTIC = 1
    MANHATTAN = 2
    RANDOM = 3

class DepotLocation(Enum):
    CENTER = 1
    OUTSKIRTS = 2
    OUTSIDE = 3

class Bin:
    def __init__(self, bin_id: int, capacity: float, fill_rate: float, pos: Tuple[float, float] = (0,0)):
        self.id = bin_id
        self.capacity = capacity
        self.initial_fill = np.random.uniform(0, capacity)
        self.fill_level = self.initial_fill
        self.fill_rate = fill_rate
        self.pos = pos  # Added position attribute

    def restore(self):
        self.fill_level = self.initial_fill

    def update_fill(self, time_elapsed: int):
        """Update the fill level based on time elapsed."""
        self.fill_level = min(self.capacity, self.fill_level + self.fill_rate * time_elapsed)

class City:
    def __init__(self, width: float, height: float, num_points: int = 100, num_bins: int = 20, 
                 city_type: CityType = CityType.REALISTIC, 
                 depot_location: DepotLocation = DepotLocation.CENTER):

        self.width = width
        self.height = height
        self.num_points = num_points
        self.num_bins = num_bins # Fixed variable name
        self.city_type = city_type
        
        self.path_matrix = None 
        self.node_order = None 
        self.graph = None
        
        # 1. Generate Graph or Grid structures
        if city_type == CityType.REALISTIC:
            self.graph = self.generate_realistic_city_graph(num_points)
            # Precompute distances
            self.path_matrix, self.node_order = self.all_pairs_shortest_path_matrix_dijkstra(self.graph)
            # Available locations are graph nodes
            possible_locations = list(self.graph.nodes())
        
        elif city_type == CityType.MANHATTAN:
            # Create a logical grid for snapping
            grid_size_x = int(np.sqrt(num_points * (width / height)))
            grid_size_y = int(np.sqrt(num_points * (height / width)))
            dx = width / grid_size_x
            dy = height / grid_size_y
            possible_locations = [
                (round(x * dx + dx/2, 2), round(y * dy + dy/2, 2)) 
                for x in range(grid_size_x) for y in range(grid_size_y)
            ]
        else:
            # Random continuous locations
            possible_locations = None 

        # 2. Set Depot Location
        self.depot = self._place_depot(depot_location, possible_locations)

        # 3. Initialize Bins
        self.bins = []
        # Exclude depot from bin locations if using discrete points
        if possible_locations:
            available_for_bins = [loc for loc in possible_locations if loc != self.depot]
            if len(available_for_bins) < num_bins:
                 # Fallback if not enough nodes: allow overlap or just take what we have
                selected_locs = random.choices(available_for_bins, k=num_bins)
            else:
                selected_locs = random.sample(available_for_bins, num_bins)
        else:
            # Pure random generation
            selected_locs = [
                (round(random.uniform(0, width), 2), round(random.uniform(0, height), 2))
                for _ in range(num_bins)
            ]

        for i in range(num_bins):
            b = Bin(
                bin_id=i, 
                capacity=random.gauss(50, 150), 
                fill_rate=0.5, # Give it some fill rate
                pos=selected_locs[i]
            )
            self.bins.append(b)

    def _place_depot(self, location_type: DepotLocation, possible_locations: List[Tuple[float, float]] = None) -> Tuple[float, float]:
        """Determines depot coordinates based on strategy and city type."""
        center = (self.width / 2, self.height / 2)
        
        target_pos = center
        if location_type == DepotLocation.OUTSKIRTS:
            target_pos = (self.width * 0.1, self.height * 0.1)
        elif location_type == DepotLocation.OUTSIDE:
            target_pos = (-self.width * 0.2, self.height / 2)

        if possible_locations:
            # Find closest valid node to the target position
            # This ensures the depot is actually on the road network
            return min(possible_locations, key=lambda p: np.linalg.norm(np.array(p) - np.array(target_pos)))
        else:
            return target_pos

    def generate_realistic_city_graph(self, num_points: int, power=2.8):
        city_radius = min(self.width, self.height) / 2

        # Dense-center sampling
        theta = np.random.uniform(0, 2*np.pi, num_points)
        r = city_radius * np.random.rand(num_points)**power
        
        # Center the coordinates relative to width/height
        cx, cy = self.width / 2, self.height / 2
        centers = np.column_stack([r*np.cos(theta) + cx, r*np.sin(theta) + cy])

        vor = Voronoi(centers)
        G = nx.Graph()

        for ridge in vor.ridge_vertices:
            if -1 in ridge:
                continue

            p1, p2 = vor.vertices[ridge[0]], vor.vertices[ridge[1]]
            
            # Check bounds
            if (0 <= p1[0] <= self.width and 0 <= p1[1] <= self.height and
                0 <= p2[0] <= self.width and 0 <= p2[1] <= self.height):
                
                u = tuple(np.round(p1, 2))
                v = tuple(np.round(p2, 2))
                dist = float(np.linalg.norm(p1 - p2))
                G.add_edge(u, v, weight=dist)

        if G.number_of_nodes() == 0:
            # Fallback simple graph if Voronoi fails
            G.add_edge((0.0,0.0), (self.width, self.height), weight=100)
            return G

        main_city = max(nx.connected_components(G), key=len)
        return G.subgraph(main_city).copy()

    def distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        if self.city_type == CityType.RANDOM:
            return np.linalg.norm(np.array(pos1) - np.array(pos2))
            
        elif self.city_type == CityType.REALISTIC:
            try:
                # Ensure tuple format for lookup
                p1 = tuple(np.round(pos1, 2))
                p2 = tuple(np.round(pos2, 2))
                
                idx1 = self.node_order.index(p1)
                idx2 = self.node_order.index(p2)
                d = self.path_matrix[idx1, idx2]
                return d if d != np.inf else 10000.0
            except ValueError:
                # Fallback if points aren't exactly on nodes (shouldn't happen with correct initialization)
                return np.linalg.norm(np.array(pos1) - np.array(pos2)) * 1.5
                
        else:
            # Manhattan
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def reset_all(self):
        for b in self.bins:
            b.restore()

    def all_pairs_shortest_path_matrix_dijkstra(self, G, weight="weight", node_order=None, dtype=float):
        if node_order is None:
            node_order = list(G.nodes())
        else:
            node_order = list(node_order)

        n = len(node_order)
        idx = {node: i for i, node in enumerate(node_order)}

        D = np.full((n, n), np.inf, dtype=dtype)
        np.fill_diagonal(D, 0.0)

        for src, dist_dict in nx.all_pairs_dijkstra_path_length(G, weight=weight):
            if src not in idx:
                continue
            i = idx[src]
            for dst, dist in dist_dict.items():
                j = idx.get(dst, None)
                if j is not None:
                    D[i, j] = float(dist)

        return D, node_order
