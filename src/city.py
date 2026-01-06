import numpy as np
from scipy.spatial import Voronoi
import random
import networkx as nx

from enum import Enum

class CityType(Enum):
    REALISTIC = 1
    MANHATTAN = 2
    RANDOM = 3

class DepotLocation(Enum):
    CENTER = 1
    OUTSKIRTS = 2
    OUTSIDE = 3

class Bin:
    def __init__(self, bin_id: int, capacity: float, fill_rate: float):
        self.id = bin_id
        self.capacity = capacity
        self.initial_fill = np.random.uniform(0, capacity)
        self.fill_level = self.initial_fill
        self.fill_rate = fill_rate

    def restore(self):
        self.fill_level = self.initial_fill

    def update_fill(self, time_elapsed: int):
        """Update the fill level based on time elapsed."""
        self.fill_level = min(self.capacity, self.fill_level + self.fill_rate * time_elapsed)

class City:
    def __init__(self, radius: float, num_district: int, number_of_bins: int, city_type: CityType = CityType.REALISTIC, depot_location: DepotLocation = DepotLocation.CENTER):
        self.radius = radius
        self.num_district = num_district
        self.number_of_bins = number_of_bins
        self.city_type = city_type
        self.bins: list[Bin] = []
    

    def generate_realistic_city_graph(self, num_points: int, power=2.8):
        city_radius = min(self.width, self.height) / 2

        # Dense-center sampling
        theta = np.random.uniform(0, 2*np.pi, num_points)
        r = city_radius * np.random.rand(num_points)**power
        centers = np.column_stack([r*np.cos(theta), r*np.sin(theta)])

        vor = Voronoi(centers)
        G = nx.Graph()

        for ridge in vor.ridge_vertices:
            if -1 in ridge:
                continue

            p1, p2 = vor.vertices[ridge[0]], vor.vertices[ridge[1]]

            if np.linalg.norm(p1) <= city_radius and np.linalg.norm(p2) <= city_radius:
                u = tuple(np.round(p1, 2))
                v = tuple(np.round(p2, 2))
                dist = float(np.linalg.norm(p1 - p2))
                G.add_edge(u, v, weight=dist)

        if G.number_of_nodes() == 0:
            return G

        main_city = max(nx.connected_components(G), key=len)
        return G.subgraph(main_city).copy()

        # Keep only the largest connected component
        main_city = max(nx.connected_components(G), key=len)
        G = G.subgraph(main_city).copy()
        return G
