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
    def __init__(self, config: dict, bin_id: int, capacity: float, pos: Tuple[float, float]):
        self.id = bin_id
        self.config = config
        self.capacity = capacity
        # Random fill between config min and 100%
        self.fill_level = np.random.uniform(capacity * self.config['bins']['min_fill_level_ratio'], capacity) 
        # Per-bin predisposition for how quickly it fills (ratio of capacity per day)
        mu = self.config['bin_refill']['base_rate_ratio']
        sigma = self.config['bin_refill']['rate_sigma_ratio']
        self.fill_rate = max(0.0, np.random.normal(mu, sigma))
        self.pos = pos

class City:
    def __init__(self,config: dict, width: float, height: float, num_points: int = 30, num_bins: int = 10, 
                 city_type: CityType = CityType.REALISTIC, distribution_type: DistributionType = DistributionType.UNIFORM):
        self.width = width
        self.height = height
        self.city_type = city_type
        self.config = config
        
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
            cap = random.choice(self.config['bins']['capacity_options']) 
            b = Bin(self.config, i, capacity=cap, pos=chosen_locs[i])
            self.bins.append(b)

        self.dist_matrix = []
        self.path_matrix = [] 
        # Mapping from POI coordinate -> matrix index (built in _precompute_poi_distances)
        self.poi_to_idx = {}

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
                u = tuple(np.round(p1, self.config['city']['coordinate_rounding']))
                v = tuple(np.round(p2, self.config['city']['coordinate_rounding']))
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
        """Calculates shortest paths between Depot and Bins into an (N+1)x(N+1) matrix."""
        # 1. Define POIs and create index mapping
        # Index 0 is always the Depot
        pois = [self.depot] + [b.pos for b in self.bins]
        num_pois = len(pois)
        
        # Create the N x N matrix initialized to infinity
        self.dist_matrix = np.full((num_pois, num_pois), np.inf)
        self.path_matrix = [[[] for _ in range(num_pois)] for _ in range(num_pois)]

        # Mapping to help translate coordinate tuples back to matrix indices
        self.poi_to_idx = {pos: i for i, pos in enumerate(pois)}
         
        print(f"Pre-computing {num_pois}x{num_pois} distance matrix...")
        
        # Note: depot and bins are chosen from valid graph nodes, so Dijkstra should work.
        # Still, be defensive in case depot isn't a graph node (e.g., graph missing / modified).
        for i, start_node in enumerate(pois):
            # Dijkstra from this specific POI to all other nodes in the road graph
            if start_node not in self.graph:
                # If a POI isn't an actual graph node, we can't run Dijkstra from it.
                # Keep row as inf; distance() will fall back to graph/euclid as needed.
                continue

            lengths, paths = nx.single_source_dijkstra(self.graph, start_node, weight='weight')
            
            for j in range(num_pois):
                end_node = pois[j]
                if end_node in lengths:
                    # Store distance    
                    self.dist_matrix[i, j] = lengths[end_node]
                    # Store the list of vertices (e.g., [(x1,y1), (x2,y2)...])
                    self.path_matrix[i][j] = paths[end_node] 
            

    def distance(self, pos1, pos2):
        """
        Retrieves the shortest path distance. 
        Uses O(1) matrix lookup for POIs, falls back to Dijkstra/Euclidean for others.
        """
        # 0. Trivial case
        if pos1 == pos2:
            return 0.0

        # 1. Check if both positions are in our POI Matrix
        if self.poi_to_idx and pos1 in self.poi_to_idx and pos2 in self.poi_to_idx:
            idx1 = self.poi_to_idx[pos1]
            idx2 = self.poi_to_idx[pos2]
            return self.dist_matrix[idx1, idx2]
        else:
            print("Positions not in POI matrix, falling back to graph distance.")
        
        # 2. Fallback: If one or both points are NOT POIs (e.g., during plotting or dynamic events)
        if self.graph:
            try:
                # Check for direct road distance
                return nx.shortest_path_length(self.graph, pos1, pos2, weight='weight')
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # If nodes aren't in the graph or no path exists, use scaled Euclidean
                return np.linalg.norm(np.array(pos1) - np.array(pos2)) * 2
        
        # 3. Last Resort: Simple straight-line distance
        return np.linalg.norm(np.array(pos1) - np.array(pos2))    


    def get_path(self, pos1, pos2):
        """Return a list of graph nodes representing the route from pos1 to pos2.

        If both endpoints are POIs (depot or bins), returns the precomputed path
        from `self.path_matrix`.

        Falls back to NetworkX shortest path when possible. If either point isn't
        a graph node, falls back to a straight line [pos1, pos2].
        """
        if pos1 == pos2:
            return [pos1]

        # Fast path: POI -> POI
        if self.poi_to_idx and pos1 in self.poi_to_idx and pos2 in self.poi_to_idx:
            i = self.poi_to_idx[pos1]
            j = self.poi_to_idx[pos2]
            path = self.path_matrix[i][j]
            if path:
                return path

        if not self.graph:
            return [pos1, pos2]

        # If positions aren't nodes, we can't run shortest_path on them.
        if pos1 not in self.graph or pos2 not in self.graph:
            return [pos1, pos2]

        try:
            return nx.shortest_path(self.graph, pos1, pos2, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [pos1, pos2]


    def generate_greedy_route(self) -> List[int]:
        """
        Generate a simple collection route using nearest-neighbor heuristic.
        Returns a list of bin IDs in visit order.
        
        This is a greedy approximation to the TSP problem - not optimal but fast.
        Time complexity: O(n²) where n is the number of bins.
        """
        if not self.bins:
            return []

        # Start from depot (POI index 0)
        current_idx = 0
        # Unvisited POI indices correspond to bins: 1..N
        unvisited = set(range(1, len(self.bins) + 1))
        route = []
        
        while unvisited:
            next_poi_idx = min(
                unvisited,
                key=lambda idx: self.dist_matrix[current_idx, idx]
            )
            
            # Store the bin ID (which is index - 1)
            route.append(next_poi_idx - 1)
            
            # Update current position and remove from unvisited
            current_idx = next_poi_idx
            unvisited.remove(next_poi_idx)
        
        return route
        
    def reset_all(self):
        pass
