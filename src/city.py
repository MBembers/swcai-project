import numpy as np

class Bin:
    def __init__(self, bin_id: int, x: float, y: float, capacity: float = 100.0):
        self.id = bin_id
        self.pos = (x, y)
        self.capacity = capacity
        # Store initial value to reset later for visualization
        self.initial_fill = np.random.randint(0, 100) 
        self.fill_level = self.initial_fill

    def restore(self):
        """Restores the bin to its initial random state."""
        self.fill_level = self.initial_fill

class CityGrid:
    def __init__(self, width: int, height: int, num_bins: int, uncertainty: float = 5.0):
        self.width = width
        self.height = height
        self.uncertainty = uncertainty
        self.bins = [
            Bin(i, np.random.uniform(5, width-5), np.random.uniform(5, height-5))
            for i in range(num_bins)
        ]
        self.depot = (width / 2, height / 2)
        
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

    def reset_all(self):
        """Resets all bins to their starting fill levels."""
        for b in self.bins:
            b.restore()
