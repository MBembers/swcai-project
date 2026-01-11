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
def plot_heatmap_comparison(city, greedy_visits, ga_visits, total_collections):
    """
    Creates side-by-side heatmaps showing visit frequency for Greedy vs GA across all collections.
    
    Args:
        city: City object with bins
        greedy_visits: dict mapping bin_id -> number of visits (Greedy)
        ga_visits: dict mapping bin_id -> number of visits (GA)
        total_collections: total number of collection runs
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Prepare data
    greedy_counts = [greedy_visits.get(b.id, 0) for b in city.bins]
    ga_counts = [ga_visits.get(b.id, 0) for b in city.bins]
    difference = [ga_counts[i] - greedy_counts[i] for i in range(len(city.bins))]
    
    bin_positions = np.array([b.pos for b in city.bins])
    
    # Plot 1: Greedy visits
    ax = axes[0]
    scatter1 = ax.scatter(bin_positions[:, 0], bin_positions[:, 1], 
                          c=greedy_counts, cmap='YlOrRd', s=150, edgecolors='black', linewidth=0.5)
    ax.scatter(*city.depot, c='blue', marker='s', s=300, label='Depot', zorder=10)
    ax.set_title(f'Greedy Method - Bin Visits\n(Total collections: {total_collections})', fontsize=12, fontweight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    cbar1 = plt.colorbar(scatter1, ax=ax)
    cbar1.set_label('Visit count')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: GA visits
    ax = axes[1]
    scatter2 = ax.scatter(bin_positions[:, 0], bin_positions[:, 1], 
                          c=ga_counts, cmap='YlOrRd', s=150, edgecolors='black', linewidth=0.5)
    ax.scatter(*city.depot, c='blue', marker='s', s=300, label='Depot', zorder=10)
    ax.set_title(f'GA Method - Bin Visits\n(Total collections: {total_collections})', fontsize=12, fontweight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    cbar2 = plt.colorbar(scatter2, ax=ax)
    cbar2.set_label('Visit count')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Difference (GA - Greedy)
    ax = axes[2]
    cmap_diff = mcolors.LinearSegmentedColormap.from_list("diff", ["red", "white", "green"])
    scatter3 = ax.scatter(bin_positions[:, 0], bin_positions[:, 1], 
                          c=difference, cmap=cmap_diff, s=150, edgecolors='black', linewidth=0.5,
                          vmin=-max(abs(min(difference)), abs(max(difference))),
                          vmax=max(abs(min(difference)), abs(max(difference))))
    ax.scatter(*city.depot, c='blue', marker='s', s=300, label='Depot', zorder=10)
    ax.set_title(f'Difference (GA - Greedy)\n(Green: GA visits more | Red: Greedy visits more)', fontsize=12, fontweight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    cbar3 = plt.colorbar(scatter3, ax=ax)
    cbar3.set_label('Visit difference')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_collection_statistics(greedy_distances, ga_distances, collection_numbers):
    """
    Plot distance trends over collection runs for both methods.
    
    Args:
        greedy_distances: list of distances for Greedy method per collection
        ga_distances: list of distances for GA method per collection
        collection_numbers: list of collection run numbers
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Distance per collection run
    ax = axes[0]
    ax.plot(collection_numbers, greedy_distances, 'o-', label='Greedy', linewidth=2, markersize=6)
    ax.plot(collection_numbers, ga_distances, 's-', label='GA', linewidth=2, markersize=6)
    ax.set_xlabel('Collection Run Number')
    ax.set_ylabel('Distance')
    ax.set_title('Route Distance Comparison Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative distance
    ax = axes[1]
    greedy_cumsum = np.cumsum(greedy_distances)
    ga_cumsum = np.cumsum(ga_distances)
    ax.plot(collection_numbers, greedy_cumsum, 'o-', label='Greedy (Cumulative)', linewidth=2, markersize=6)
    ax.plot(collection_numbers, ga_cumsum, 's-', label='GA (Cumulative)', linewidth=2, markersize=6)
    ax.fill_between(collection_numbers, greedy_cumsum, ga_cumsum, alpha=0.2)
    ax.set_xlabel('Collection Run Number')
    ax.set_ylabel('Cumulative Distance')
    ax.set_title('Cumulative Distance Over Year')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def _get_route_edges(city, route_ids):
    """
    Extract edges (bin-to-bin connections) from a route.
    Returns a set of edge tuples sorted to handle bidirectionality.
    """
    edges = set()
    current_pos = city.depot
    
    for bin_id in route_ids:
        bin_pos = city.bins[bin_id].pos
        # Create normalized edge (smaller first for consistency)
        edge = tuple(sorted([current_pos, bin_pos]))
        edges.add(edge)
        current_pos = bin_pos
    
    # Return to depot
    edge = tuple(sorted([current_pos, city.depot]))
    edges.add(edge)
    
    return edges

def plot_route_comparison(city, greedy_route, ga_route, collection_num):
    """
    Show how GA modifies the Greedy route:
    - Red paths: removed by GA (in Greedy but not in GA)
    - Green paths: added by GA (in GA but not in Greedy)
    - Gray paths: common to both routes
    
    Args:
        city: City object
        greedy_route: list of bin IDs for greedy route
        ga_route: list of bin IDs for GA route
        collection_num: collection run number (for title)
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw road network
    if city.graph:
        pos = {n: n for n in city.graph.nodes()}
        nx.draw_networkx_edges(city.graph, pos, edge_color="#444444", 
                               width=0.8, alpha=0.6, ax=ax)
    
    # Get route edges
    greedy_edges = _get_route_edges(city, greedy_route)
    ga_edges = _get_route_edges(city, ga_route)
    
    # Categorize edges
    removed_edges = greedy_edges - ga_edges      # Red: in Greedy, not in GA
    added_edges = ga_edges - greedy_edges        # Green: in GA, not in Greedy
    common_edges = greedy_edges & ga_edges       # Gray: in both
    
    # Draw removed edges (Red)
    for (pos1, pos2) in removed_edges:
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                color='red', linewidth=3, alpha=0.7, label='Removed' if (pos1, pos2) == list(removed_edges)[0] else '')
    
    # Draw added edges (Green)
    for (pos1, pos2) in added_edges:
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                color='green', linewidth=3, alpha=0.7, label='Added' if (pos1, pos2) == list(added_edges)[0] else '')
    
    # Draw common edges (Gray)
    for (pos1, pos2) in common_edges:
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                color='gray', linewidth=2, alpha=0.5, linestyle='--', label='Common' if (pos1, pos2) == list(common_edges)[0] else '')
    
    # Draw bins
    cmap = mcolors.LinearSegmentedColormap.from_list("waste", ["#2ecc71", "#f1c40f", "#e74c3c"])
    for b in city.bins:
        fill_ratio = b.fill_level / b.capacity
        color = cmap(max(0, min(1, fill_ratio)))
        ax.scatter(b.pos[0], b.pos[1], s=150, color=color, edgecolors='black', linewidth=0.5, zorder=5)
    
    # Draw depot
    ax.scatter(*city.depot, c='blue', marker='s', s=300, label='Depot', zorder=10, edgecolors='black', linewidth=1)
    
    # Statistics
    ax.set_title(f'Collection Run #{collection_num}: Route Changes (GA vs Greedy)\n' + 
                 f'Removed: {len(removed_edges)} edges | Added: {len(added_edges)} edges | Common: {len(common_edges)} edges',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True, alpha=0.2)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=3, label=f'Removed ({len(removed_edges)} edges)'),
        Line2D([0], [0], color='green', linewidth=3, label=f'Added ({len(added_edges)} edges)'),
        Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label=f'Common ({len(common_edges)} edges)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, label='Depot'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot_aggregate_route_changes(city, all_removed_edges, all_added_edges, all_common_edges):
    """
    Show aggregate route changes across all collection runs.
    Edge thickness represents frequency of change.
    
    Args:
        city: City object
        all_removed_edges: dict mapping edge -> count of times removed
        all_added_edges: dict mapping edge -> count of times added
        all_common_edges: dict mapping edge -> count of times used in common
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Draw road network
    if city.graph:
        pos = {n: n for n in city.graph.nodes()}
        nx.draw_networkx_edges(city.graph, pos, edge_color="#333333", 
                               width=0.6, alpha=0.5, ax=ax)
    
    # Normalize edge counts for line width
    max_removed = max(all_removed_edges.values()) if all_removed_edges else 1
    max_added = max(all_added_edges.values()) if all_added_edges else 1
    max_common = max(all_common_edges.values()) if all_common_edges else 1
    
    # Draw removed edges (Red) - thicker = more often removed
    for (pos1, pos2), count in all_removed_edges.items():
        linewidth = 1 + (count / max_removed) * 3
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                color='red', linewidth=linewidth, alpha=0.6)
    
    # Draw added edges (Green) - thicker = more often added
    for (pos1, pos2), count in all_added_edges.items():
        linewidth = 1 + (count / max_added) * 3
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                color='green', linewidth=linewidth, alpha=0.6)
    
    # Draw common edges (Gray) - thicker = more often used together
    for (pos1, pos2), count in all_common_edges.items():
        linewidth = 0.5 + (count / max_common) * 2
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                color='gray', linewidth=linewidth, alpha=0.3, linestyle='--')
    
    # Draw bins
    cmap = mcolors.LinearSegmentedColormap.from_list("waste", ["#2ecc71", "#f1c40f", "#e74c3c"])
    for b in city.bins:
        fill_ratio = b.fill_level / b.capacity
        color = cmap(max(0, min(1, fill_ratio)))
        ax.scatter(b.pos[0], b.pos[1], s=150, color=color, edgecolors='black', linewidth=0.5, zorder=5)
    
    # Draw depot
    ax.scatter(*city.depot, c='blue', marker='s', s=300, label='Depot', zorder=10, edgecolors='black', linewidth=1)
    
    ax.set_title(f'Aggregate Route Changes Over All Collections\n(Edge thickness = frequency of change)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True, alpha=0.2)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Removed by GA'),
        Line2D([0], [0], color='green', linewidth=2, label='Added by GA'),
        Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='Common routes'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, label='Depot'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()