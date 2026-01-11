import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
from itertools import cycle
from .config import CONFIG

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
    print("Preparing visualization...")
    plt.figure(figsize=tuple(CONFIG['visualization']['figure_size']))
    
    # 1. Background Road Network
    if city.graph:
        pos = {n: n for n in city.graph.nodes()}
        nx.draw_networkx_edges(city.graph, pos, edge_color="#555555", 
                               width=CONFIG['visualization']['edge_width'], 
                               alpha=CONFIG['visualization']['edge_alpha'])

    # 2. Expert/Agent Layer: Bins with Load Spectrum
    # Color Map: Green (Empty) -> Yellow -> Red (Full)
    cmap = mcolors.LinearSegmentedColormap.from_list("waste", ["#2ecc71", "#f1c40f", "#e74c3c"])
    
    for b in city.bins:
        fill_ratio = b.fill_level / b.capacity
        color = cmap(max(0, min(1, fill_ratio)))
        plt.scatter(b.pos[0], b.pos[1], s=CONFIG['visualization']['bin_scatter_size'], 
                    color=color, edgecolors='black', zorder=CONFIG['visualization']['bin_zorder'])
        plt.text(b.pos[0], b.pos[1]+CONFIG['visualization']['bin_text_offset'], f"{int(fill_ratio*100)}%", 
                fontsize=CONFIG['visualization']['bin_font_size'], ha='center')

    # 3. GA/Agent Layer: Routes (Road-Following)
    segments = sim.get_trip_segments(route_ids)
    colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c']
    
    for i, trip in enumerate(segments):
        trip_color = colors[i % len(colors)]
        for j in range(len(trip)-1):
            try:
                # Get the actual road nodes between targets
                path_nodes = nx.shortest_path(city.graph, trip[j], trip[j+1], weight='weight')
                xs, ys = zip(*path_nodes)
                plt.plot(xs, ys, color=trip_color, linewidth=CONFIG['visualization']['route_line_width'], 
                        alpha=CONFIG['visualization']['route_alpha'], zorder=CONFIG['visualization']['route_zorder'])
                
                # Draw a single arrow if the path is long enough
                if len(xs) > 2:
                    mid = len(xs) // 2
                    plt.annotate('', xy=(xs[mid+1], ys[mid+1]), xytext=(xs[mid], ys[mid]),
                                 arrowprops=dict(arrowstyle='->', color=trip_color))
            except:
                plt.plot([trip[j][0], trip[j+1][0]], [trip[j][1], trip[j+1][1]], 
                         color=trip_color, linestyle='--', alpha=0.3)

    plt.scatter(*city.depot, c='black', marker='s', s=CONFIG['visualization']['depot_marker_size'], 
                label='Depot', zorder=CONFIG['visualization']['depot_zorder'])
    plt.title("Smart Waste Routing: Agent Paths Following Road Network")
    plt.show()
