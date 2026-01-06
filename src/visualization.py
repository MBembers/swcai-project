import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
from itertools import cycle

def get_actual_path_coords(city, start_pos, end_pos):
    """
    Finds the specific sequence of coordinates following the road network
    between two points.
    """
    if not city.graph:
        return [start_pos, end_pos]

    # Find closest graph nodes to the coordinate positions
    # (In the new city implementation, bin.pos are already nodes, but this is safer)
    try:
        start_node = min(city.graph.nodes(), key=lambda n: city.distance(n, start_pos))
        end_node = min(city.graph.nodes(), key=lambda n: city.distance(n, end_pos))
        
        path_nodes = nx.shortest_path(city.graph, start_node, end_node, weight='weight')
        return path_nodes # List of tuples (x, y)
    except (nx.NetworkXNoPath, ValueError):
        return [start_pos, end_pos] # Fallback to straight line

def plot_simulation(city, truck, route_ids, sim):
    plt.figure(figsize=(15, 15))
    
    # 1. Draw Road Network (Background) - Lighter/Thinner for visibility
    if city.graph:
        pos = {n: n for n in city.graph.nodes()}
        nx.draw_networkx_edges(
            city.graph, pos, 
            edge_color='#e0e0e0', 
            width=0.8, 
            alpha=0.6
        )

    # 2. Draw Depot
    plt.scatter(*city.depot, c='black', marker='s', s=300, label='Depot', zorder=10)

    # 3. Draw Routes (Actual Paths)
    segments = sim.get_trip_segments(route_ids)
    
    # Distinct colors for different routes
    colors = cycle(['#1f77b4', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22'])
    
    for i, trip_targets in enumerate(segments):
        if len(trip_targets) < 2: continue
        
        trip_color = next(colors)
        
        # We need to plot the path between each target in the trip
        for j in range(len(trip_targets) - 1):
            start = trip_targets[j]
            end = trip_targets[j+1]
            
            # Get the winding road path
            path_coords = get_actual_path_coords(city, start, end)
            xs, ys = zip(*path_coords)
            
            # Plot the line
            plt.plot(xs, ys, c=trip_color, linewidth=2.5, alpha=0.8, zorder=5)
            
            # Add small arrow at the midpoint of this specific segment
            mid = len(xs) // 2
            if mid > 0:
                plt.arrow(xs[mid], ys[mid], 
                          xs[mid+1]-xs[mid], ys[mid+1]-ys[mid], 
                          shape='full', lw=0, length_includes_head=True, 
                          head_width=2, color=trip_color, zorder=6)

    # 4. Draw Bins (On top of lines)
    # Color Map: Green (Empty) -> Yellow -> Red (Full)
    cmap = mcolors.LinearSegmentedColormap.from_list("waste_load", ["#00ff00", "#ffff00", "#ff0000"])

    for b in city.bins:
        fill_ratio = b.fill_level / b.capacity
        # Clamp between 0 and 1
        fill_ratio = max(0.0, min(1.0, fill_ratio))
        
        color = cmap(fill_ratio)
        
        # Draw Bin Circle
        plt.scatter(b.pos[0], b.pos[1], s=120, color=color, edgecolors='#444444', linewidth=1, zorder=20)
        
        # Draw Percentage Text
        plt.text(b.pos[0], b.pos[1], f"{int(fill_ratio*100)}%", 
                 ha='center', va='center', fontsize=7, fontweight='bold', color='black', zorder=21)

    plt.title(f"Optimized Waste Collection: {len(segments)} Trips", fontsize=16)
    plt.axis('equal') # Keep aspect ratio square so roads don't look squashed
    plt.tight_layout()
    plt.show()
