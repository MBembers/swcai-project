import numpy as np
from scipy.spatial import Voronoi
import random
import networkx as nx
import math
from enum import Enum
from typing import Tuple, List, Dict

class CityType(Enum):
    REALISTIC = 1
    MANHATTAN = 2
    RANDOM = 3

class DistributionType(Enum):
    UNIFORM = 1
    EXPONENTIAL_DECAY = 2

class Bin:
    def __init__(self, bin_id: int, capacity: float, pos: Tuple[float, float]):
        self.id = bin_id
        self.capacity = capacity
        # Random fill between 5% and 100%
        self.fill_level = np.random.uniform(capacity * 0.05, capacity) 
        self.pos = pos

class City:
    def __init__(self, width: float, height: float, num_points: int = 30, num_bins: int = 10, 
                 city_type: CityType = CityType.REALISTIC, distribution_type: DistributionType = DistributionType.UNIFORM):
        self.width = width
        self.height = height
        self.city_type = city_type
        
        # 1. Generate Graph
        if city_type == CityType.REALISTIC:
            self.graph = self._generate_voronoi_city(num_points)
            self.valid_nodes = list(self.graph.nodes())
        elif city_type == CityType.MANHATTAN:
            self.graph = self._generate_manhattan_city(num_points)
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
        elif (distribution_type == DistributionType.UNIFORM):
            chosen_locs = random.sample(available_locs, num_bins)
        elif (distribution_type == DistributionType.EXPONENTIAL_DECAY):
            locs = np.array(available_locs, dtype = float)
            center = np.array([self.width / 2.0, self.height / 2.0])
            d = np.linalg.norm(locs - center, axis=1)
            # Exponential decay scale
            decay_scale = 0.30 * max(self.width, self.height)
            decay_scale = max(decay_scale, 1e-9)
            weights = np.exp(-d / decay_scale)
            probs = weights / weights.sum()
            idx = np.random.choice(len(available_locs), size=num_bins, replace=False, p=probs)
            chosen_locs = [available_locs[i] for i in idx]

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
                if u != v:
                    G.add_edge(u, v, weight=dist)

        # Return largest connected component
        if len(G) > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            return G.subgraph(largest_cc).copy()
        return G
    
    def _generate_manhattan_city(self, num_points) -> nx.Graph:
        if num_points < 2:
            return nx.Graph()

        W, H = float(self.width), float(self.height)

        # Choose grid dimensions to match aspect ratio and target num_points
        aspect = W / H if H > 0 else 1.0
        nx_count = max(2, int(math.ceil(math.sqrt(num_points * aspect))))
        ny_count = max(2, int(math.ceil(num_points / nx_count)))

        # Compute spacing so grid fits within width/height
        sx = W / (nx_count - 1) if nx_count > 1 else 0.0
        sy = H / (ny_count - 1) if ny_count > 1 else 0.0

        # Build full integer grid first
        G_full = nx.grid_2d_graph(nx_count, ny_count)  # nodes are (i, j)

        # Map integer grid nodes -> (x, y) coordinates
        mapping = {}
        for (i, j) in G_full.nodes():
            x = float(i * sx)
            y = float(j * sy)
            mapping[(i, j)] = (x, y)

        G_full = nx.relabel_nodes(G_full, mapping)

        # Keep exactly num_points nodes 
        nodes_sorted = sorted(G_full.nodes(), key=lambda p: (p[0], p[1]))
        keep_nodes = nodes_sorted[:num_points]
        G = G_full.subgraph(keep_nodes).copy()

        # Edge weights
        for u, v in G.edges():
            G[u][v]["weight"] = float(np.hypot(u[0] - v[0], u[1] - v[1]))

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
    
    def generate_greedy_route(self) -> List[int]:
        """
        Generate a simple collection route using nearest-neighbor heuristic.
        Returns a list of bin IDs in visit order.
        
        This is a greedy approximation to the TSP problem - not optimal but fast.
        Time complexity: O(n²) where n is the number of bins.
        """
        if not self.bins:
            return []
        
        # Start from depot
        current_pos = self.depot
        unvisited = set(range(len(self.bins)))
        route = []
        
        while unvisited:
            # Find nearest unvisited bin
            nearest_bin_id = min(
                unvisited,
                key=lambda bid: self.distance(current_pos, self.bins[bid].pos)
            )
            
            route.append(nearest_bin_id)
            current_pos = self.bins[nearest_bin_id].pos
            unvisited.remove(nearest_bin_id)
        
        return route

    def reset_all(self):
        pass
