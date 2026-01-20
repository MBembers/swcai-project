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

    try:
        # Use straight-line distance to snap arbitrary coordinates to nearest graph nodes.
        # (city.distance falls back to graph shortest paths which expects node inputs.)
        start_node = min(city.graph.nodes(), key=lambda n: np.linalg.norm(np.array(n) - np.array(start_pos)))
        end_node = min(city.graph.nodes(), key=lambda n: np.linalg.norm(np.array(n) - np.array(end_pos)))
        
        path_nodes = nx.shortest_path(city.graph, start_node, end_node, weight='weight')
        return path_nodes 
    except (nx.NetworkXNoPath, ValueError):
        print("No path found between", start_pos, "and", end_pos)
        return [start_pos, end_pos]

def plot_simulation(city, CONFIG, truck, route_ids, sim):
    print("Preparing visualization...")
    fig = plt.figure(figsize=tuple(CONFIG['visualization']['figure_size']))
    
    if city.graph:
        pos = {n: n for n in city.graph.nodes()}
        nx.draw_networkx_edges(city.graph, pos, edge_color="#555555", 
                               width=CONFIG['visualization']['edge_width'], 
                               alpha=CONFIG['visualization']['edge_alpha'])

    cmap = mcolors.LinearSegmentedColormap.from_list("waste", ["#2ecc71", "#f1c40f", "#e74c3c"])
    
    for b in city.bins:
        fill_ratio = b.fill_level / b.capacity
        color = cmap(max(0, min(1, fill_ratio)))
        plt.scatter(b.pos[0], b.pos[1], s=CONFIG['visualization']['bin_scatter_size'], 
                    color=color, edgecolors='black', zorder=CONFIG['visualization']['bin_zorder'])
        plt.text(b.pos[0], b.pos[1]+CONFIG['visualization']['bin_text_offset'], f"{int(fill_ratio*100)}%", 
                fontsize=CONFIG['visualization']['bin_font_size'], ha='center')

    segments = sim.get_trip_segments(route_ids)
    colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c']
    
    for i, trip in enumerate(segments):
        trip_color = colors[i % len(colors)]
        for j in range(len(trip)-1):
            try:
                path_nodes = city.get_path(trip[j], trip[j+1])
                xs, ys = zip(*path_nodes)
                plt.plot(xs, ys, color=trip_color, linewidth=CONFIG['visualization']['route_line_width'], 
                        alpha=CONFIG['visualization']['route_alpha'], zorder=CONFIG['visualization']['route_zorder'])
                
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
    return fig

def plot_heatmap_comparison(city, greedy_visits, ga_visits, total_collections):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
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
    cbar1 = fig.colorbar(scatter1, ax=ax)
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
    cbar2 = fig.colorbar(scatter2, ax=ax)
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
    cbar3 = fig.colorbar(scatter3, ax=ax)
    cbar3.set_label('Visit difference')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_collection_statistics(greedy_distances, ga_distances, collection_numbers):
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
    return fig



def _normalize_edge(u, v):
    """Return an undirected edge key for counting/sets."""
    return (u, v) if u <= v else (v, u)


def _expand_route_nodes(city, route_ids):
    """Expand a POI route (bin IDs) into a full list of graph nodes.

    Uses `city.get_path()` which prefers the precomputed `city.path_matrix`.
    """
    full_nodes = []
    current_pos = city.depot

    for bin_id in route_ids:
        next_pos = city.bins[bin_id].pos
        leg = city.get_path(current_pos, next_pos)
        if not leg:
            leg = [current_pos, next_pos]

        if not full_nodes:
            full_nodes.extend(leg)
        else:
            # Avoid duplicating the joint node
            full_nodes.extend(leg[1:] if leg[0] == full_nodes[-1] else leg)

        current_pos = next_pos

    # Return to depot
    leg = city.get_path(current_pos, city.depot)
    if not leg:
        leg = [current_pos, city.depot]
    if not full_nodes:
        full_nodes.extend(leg)
    else:
        full_nodes.extend(leg[1:] if leg[0] == full_nodes[-1] else leg)

    return full_nodes


def _get_route_edges(city, route_ids):
    """Return a set of road-network edges actually traversed by the route."""
    nodes = _expand_route_nodes(city, route_ids)
    edges = set()
    for u, v in zip(nodes, nodes[1:]):
        edges.add(_normalize_edge(u, v))
    return edges

def plot_route_comparison(city, greedy_route, ga_route, collection_num):
    fig, ax = plt.subplots(figsize=(12, 12))
    
    if city.graph:
        pos = {n: n for n in city.graph.nodes()}
        nx.draw_networkx_edges(city.graph, pos, edge_color="#444444", 
                               width=0.8, alpha=0.6, ax=ax)
    
    greedy_edges = _get_route_edges(city, greedy_route)
    ga_edges = _get_route_edges(city, ga_route)
    
    removed_edges = greedy_edges - ga_edges
    added_edges = ga_edges - greedy_edges
    common_edges = greedy_edges & ga_edges
    
    for (pos1, pos2) in removed_edges:
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color='red', linewidth=3, alpha=0.7)

    for (pos1, pos2) in added_edges:
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color='green', linewidth=3, alpha=0.7)

    for (pos1, pos2) in common_edges:
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color='gray', linewidth=2, alpha=0.5, linestyle='--')
    
    cmap = mcolors.LinearSegmentedColormap.from_list("waste", ["#2ecc71", "#f1c40f", "#e74c3c"])
    for b in city.bins:
        fill_ratio = b.fill_level / b.capacity
        color = cmap(max(0, min(1, fill_ratio)))
        ax.scatter(b.pos[0], b.pos[1], s=150, color=color, edgecolors='black', linewidth=0.5, zorder=5)
    
    ax.scatter(*city.depot, c='blue', marker='s', s=300, label='Depot', zorder=10, edgecolors='black', linewidth=1)
    
    ax.set_title(f'Collection Run #{collection_num}: Route Changes (GA vs Greedy)\n' + 
                 f'Removed: {len(removed_edges)} edges | Added: {len(added_edges)} edges | Common: {len(common_edges)} edges',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True, alpha=0.2)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=3, label=f'Removed ({len(removed_edges)})'),
        Line2D([0], [0], color='green', linewidth=3, label=f'Added ({len(added_edges)})'),
        Line2D([0], [0], color='darkblue', linewidth=2, linestyle='--', label=f'Common ({len(common_edges)})'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, label='Depot'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig

def plot_aggregate_route_changes(city, all_removed_edges, all_added_edges, all_common_edges):
    fig, ax = plt.subplots(figsize=(14, 14))
    
    if city.graph:
        pos = {n: n for n in city.graph.nodes()}
        nx.draw_networkx_edges(city.graph, pos, edge_color="#333333", 
                               width=0.6, alpha=0.5, ax=ax)
    
    max_removed = max(all_removed_edges.values()) if all_removed_edges else 1
    max_added = max(all_added_edges.values()) if all_added_edges else 1
    max_common = max(all_common_edges.values()) if all_common_edges else 1
    
    for (pos1, pos2), count in all_removed_edges.items():
        linewidth = 1 + (count / max_removed) * 3
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color='red', linewidth=linewidth, alpha=0.6)
    
    for (pos1, pos2), count in all_added_edges.items():
        linewidth = 1 + (count / max_added) * 3
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color='green', linewidth=linewidth, alpha=0.6)
    
    for (pos1, pos2), count in all_common_edges.items():
        linewidth = 0.5 + (count / max_common) * 2
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color='darkblue', linewidth=linewidth, alpha=0.3, linestyle='--')
    
    cmap = mcolors.LinearSegmentedColormap.from_list("waste", ["#2ecc71", "#f1c40f", "#e74c3c"])
    for b in city.bins:
        fill_ratio = b.fill_level / b.capacity
        color = cmap(max(0, min(1, fill_ratio)))
        ax.scatter(b.pos[0], b.pos[1], s=150, color=color, edgecolors='black', linewidth=0.5, zorder=5)
    
    ax.scatter(*city.depot, c='blue', marker='s', s=300, label='Depot', zorder=10, edgecolors='black', linewidth=1)
    
    ax.set_title(f'Aggregate Route Changes Over All Collections\n(Edge thickness = frequency of change)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True, alpha=0.2)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Removed by GA'),
        Line2D([0], [0], color='green', linewidth=2, label='Added by GA'),
        Line2D([0], [0], color='darkblue', linewidth=2, linestyle='--', label='Common routes'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, label='Depot'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig

def plot_greedy_vs_ga_times_per_day(greedy_times, ga_times, total_days):
    days = list(range(1, len(greedy_times) + 1))
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(days, greedy_times, 'o-', label='Greedy Method', linewidth=2, markersize=6)
    plt.plot(days, ga_times, 's-', label='GA Method', linewidth=2, markersize=6)
    
    plt.xlabel('Collection Day')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Computation Time Comparison: Greedy vs GA Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
