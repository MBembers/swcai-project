import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from .city import CityGrid
from .agents import Truck
from .simulation import Simulation

def print_route_details(city: CityGrid, truck: Truck, route_ids: list[int], sim: Simulation):
    """Prints the precise sequence of bins for each trip."""
    print("\n" + "="*40)
    print("       OPTIMIZED ROUTE DETAILS       ")
    print("="*40)
    
    current_load = 0.0
    trip_count = 1
    current_trip_bins = []

    for b_id in route_ids:
        bin_obj = city.bins[b_id]
        
        # Skip empty bins
        if bin_obj.fill_level < 5:
            continue
            
        # Check if full -> New Trip
        if current_load + bin_obj.fill_level > truck.capacity:
            print(f"TRIP {trip_count}: Depot -> {current_trip_bins} -> Depot (Load: {int(current_load)})")
            trip_count += 1
            current_load = 0.0
            current_trip_bins = []
            
        current_trip_bins.append(bin_obj.id)
        current_load += bin_obj.fill_level

    # Print final trip
    if current_trip_bins:
        print(f"TRIP {trip_count}: Depot -> {current_trip_bins} -> Depot (Load: {int(current_load)})")
    print("="*40 + "\n")

def plot_simulation(city: CityGrid, truck: Truck, route_ids: list[int], sim: Simulation):
    # Print the route details first
    print_route_details(city, truck, route_ids, sim)

    plt.figure(figsize=(10, 10))
    
    # 1. Plot Bins
    cmap = mcolors.LinearSegmentedColormap.from_list("waste", ["lightgreen", "gold", "red"])
    
    for b in city.bins:
        fill_ratio = min(1.0, b.fill_level / 100.0)
        color = cmap(fill_ratio)
        plt.scatter(b.pos[0], b.pos[1], s=150, color=color, edgecolors='black', zorder=2)
        plt.text(b.pos[0], b.pos[1]-4, f"{int(b.fill_level)}%", ha='center', fontsize=8, fontweight='bold')
        # Add ID label
        plt.text(b.pos[0], b.pos[1]+4, f"ID:{b.id}", ha='center', fontsize=8, color='blue')

    # 2. Plot Depot
    plt.scatter(*city.depot, c='black', marker='s', s=200, label='Depot', zorder=3)

    # 3. Plot Routes
    segments = sim.get_trip_segments(route_ids)
    colors = ['blue', 'purple', 'orange', 'cyan', 'magenta', 'brown'] 
    
    for i, seg in enumerate(segments):
        if len(seg) < 2: continue

        xs, ys = zip(*seg)
        c = colors[i % len(colors)]
        
        plt.plot(xs, ys, color=c, linestyle='--', linewidth=2, alpha=0.6, label=f'Trip {i+1}')
        
        # Arrow Logic
        mid_idx = (len(seg) - 1) // 2
        p1 = seg[mid_idx]
        p2 = seg[mid_idx + 1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        plt.arrow(p1[0], p1[1], dx * 0.5, dy * 0.5, head_width=2, color=c, length_includes_head=True, zorder=4)

    plt.title(f"Optimization Result: {len(segments)} Trips")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.show()
