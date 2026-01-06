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

    def reset_all(self):
        """Resets all bins to their starting fill levels."""
        for b in self.bins:
            b.restore()
