import numpy as np
from scipy.spatial import Voronoi
import random
import networkx as nx
from enum import Enum
from typing import Tuple, List, Dict

class CityType(Enum):
    REALISTIC = 1
    MANHATTAN = 2
    RANDOM = 3

class Bin:
    def __init__(self, bin_id: int, capacity: float, pos: Tuple[float, float]):
        self.id = bin_id
        self.capacity = capacity
        # Random fill between 5% and 100%
        self.fill_level = np.random.uniform(capacity * 0.05, capacity) 
        self.pos = pos

class City:
    def __init__(self, width: float, height: float, num_points: int = 100, num_bins: int = 30, 
                 city_type: CityType = CityType.REALISTIC):
        self.width = width
        self.height = height
        self.city_type = city_type
        
        # 1. Generate Graph
        if city_type == CityType.REALISTIC:
            self.graph = self._generate_voronoi_city(num_points)
            self.valid_nodes = list(self.graph.nodes())
        else:
            self.graph = None
            self.valid_nodes = []

        # 2. Place Depot (Central node)
        if self.valid_nodes:
            # Find node closest to true center
            center = (width/2, height/2)
            self.depot = min(self.valid_nodes, key=lambda n: np.linalg.norm(np.array(n)-np.array(center)))
        else:
            self.depot = (width/2, height/2)

        # 3. Place Bins
        self.bins = []
        # Filter nodes to ensure bins aren't ON the depot
        available_locs = [n for n in self.valid_nodes if n != self.depot]
        
        if len(available_locs) < num_bins:
            # Fallback if graph is too small (shouldn't happen with 1000 points)
            chosen_locs = random.choices(available_locs, k=num_bins)
        else:
            chosen_locs = random.sample(available_locs, num_bins)

        for i in range(num_bins):
            # Varied capacity for realism
            cap = random.choice([50, 100, 200]) 
            b = Bin(i, capacity=cap, pos=chosen_locs[i])
            self.bins.append(b)

        # 4. Precompute Distances (Crucial for Speed)
        self.dist_cache = {}
        if self.graph:
            print("Pre-computing city distances (this may take a moment)...")
            self._precompute_poi_distances()

    def _generate_voronoi_city(self, num_points):
        # Increased randomness for organic look
        points = np.random.rand(num_points, 2) * np.array([self.width, self.height])
        vor = Voronoi(points)
        G = nx.Graph()

        for p1_idx, p2_idx in vor.ridge_vertices:
            if p1_idx == -1 or p2_idx == -1: continue
            p1, p2 = vor.vertices[p1_idx], vor.vertices[p2_idx]
            
            if (0 <= p1[0] <= self.width and 0 <= p1[1] <= self.height and
                0 <= p2[0] <= self.width and 0 <= p2[1] <= self.height):
                
                # Rounding keys is crucial for matching coords later
                u = tuple(np.round(p1, 1))
                v = tuple(np.round(p2, 1))
                dist = np.linalg.norm(p1 - p2)
                G.add_edge(u, v, weight=dist)

        # Return largest connected component
        if len(G) > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            return G.subgraph(largest_cc).copy()
        return G

    def _precompute_poi_distances(self):
        """Calculates shortest paths ONLY between Depot and Bins, and Bin to Bin."""
        # Points of Interest: Depot + All Bins
        pois = [self.depot] + [b.pos for b in self.bins]
        
        # We use multi-source dijkstra for efficiency
        # This calculates distance from one source to ALL other nodes in the graph
        # We just save the ones that matter (other POIs)
        
        count = 0
        total = len(pois)
        
        for start_node in pois:
            # Dijkstra returns dict: {target_node: distance}
            length_dict = nx.single_source_dijkstra_path_length(self.graph, start_node, weight='weight')
            
            for end_node in pois:
                if end_node in length_dict:
                    self.dist_cache[(start_node, end_node)] = length_dict[end_node]
            
            count += 1
            if count % 50 == 0:
                print(f"  Mapped {count}/{total} nodes...")

    def distance(self, pos1, pos2):
        # 1. Check Cache
        if (pos1, pos2) in self.dist_cache:
            return self.dist_cache[(pos1, pos2)]
        
        # 2. Symmetrical check
        if (pos2, pos1) in self.dist_cache:
            return self.dist_cache[(pos2, pos1)]

        # 3. Fallback (Slow, used for visualization plotting only)
        if self.graph:
            try:
                return nx.shortest_path_length(self.graph, pos1, pos2, weight='weight')
            except:
                return np.linalg.norm(np.array(pos1) - np.array(pos2)) * 2
        
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def reset_all(self):
        pass
