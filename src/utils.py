import numpy as np

def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Calculates the Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

