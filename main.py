import os
from src.seed import init_seed
from src.city import City, CityType, DistributionType
from src.agents import Truck
from src.simulation import Simulation
from src.evolution import GeneticOptimizer
from src.visualization import plot_simulation, plot_heatmap_comparison, plot_collection_statistics, plot_route_comparison, plot_aggregate_route_changes, plot_greedy_vs_ga_times_per_day
from src.expert_rules import Action
from src.utils import log_time
import time

import argparse
import src.config as config_module

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="Config file (relative to data/ or absolute path)",
    )
    return parser.parse_args()

args = parse_args()
if args.config and not os.path.isabs(args.config):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "data", args.config)
else:
    config_path = args.config

config_module.CONFIG = config_module.load_config(config_path)
print(f'Loaded config from {config_path}')

CONFIG = config_module.CONFIG  # local alias (safe now)

init_seed(CONFIG["run"]["seed"])

# ===== 1. Setup City (Scaled Up) =====
# Increased num_points to 1200 to accommodate 300 bins comfortably
print("Generating City Graph...")
city = City(
    config=CONFIG,
    width=CONFIG['city']['width'], 
    height=CONFIG['city']['height'], 
    num_points=CONFIG['city']['num_points'], 
    num_bins=CONFIG['city']['num_bins'],
    city_type=CityType[CONFIG['city']['city_type']],
    distribution_type=DistributionType[CONFIG['city']['distribution_type']]
)

# ===== 2. Truck (Scaled Up) =====
truck = Truck(
    truck_id=1,
    start_pos=city.depot,
    capacity=CONFIG['truck']['capacity']
)

sim = Simulation(CONFIG, city, truck)

# ===== 3. Multi-day Simulation =====
total_days = CONFIG['simulation']['days']
interval = CONFIG['simulation']['collection_interval_days']

print(f"Running simulation for {total_days} days, collections every {interval} days...")

greedy_total_distance = 0.0
ga_total_distance = 0.0
collection_count = 0

# Track visit frequencies for heatmap
greedy_visits = {}
ga_visits = {}
for b in city.bins:
    greedy_visits[b.id] = 0
    ga_visits[b.id] = 0

greedy_distances = []
ga_distances = []
collection_numbers = []

# Track route edge changes for aggregate visualization
all_removed_edges = {}
all_added_edges = {}
all_common_edges = {}

last_collection_num = 0
last_greedy_route = None
last_ga_route = None


start = log_time("Simulation Total", time.perf_counter())

greedy_times = []
ga_times = []

for day in range(1, total_days + 1):
    sim.refill_bins(days=1)

    if day % interval != 0:
        continue

    collection_count += 1
    print(f"\n{'='*70}")
    print(f"Day {day}: Collection Run #{collection_count}")
    print(f"{'='*70}")
    
    # Sample observations ONCE for this collection cycle
    sim.sample_collection_observations()
    
    # Use learned rates to choose bins likely to be worth visiting by next collection
    active_bin_ids = sim.plan_active_bins(collection_interval_days=interval)
    print(f"Active bins for this run: {len(active_bin_ids)} / {len(city.bins)}")

    # Baseline greedy on the active set

    start_greedy = time.perf_counter() # timer greedy

    greedy_full = city.generate_greedy_route()
    active_set = set(active_bin_ids)
    greedy_route = [bid for bid in greedy_full if bid in active_set]
    greedy_dist, _, greedy_collected = sim.simulate_route(greedy_route)
    print(f"Greedy -> distance: {greedy_dist:8.2f} | bins collected: {greedy_collected:3d}")
    greedy_total_distance += greedy_dist
    greedy_distances.append(greedy_dist)
    
    # Track greedy visits
    for bin_id in greedy_route[:greedy_collected]:
        greedy_visits[bin_id] += 1


    end_greedy = time.perf_counter()
    greedy_times.append(end_greedy - start_greedy)



    start_ga = time.perf_counter() # timer ga

    # Optimize with GA starting from greedy
    optimizer = GeneticOptimizer(
        CONFIG,
        active_bin_ids,
        sim.get_fitness,
        pop_size=CONFIG['evolution']['pop_size'],
        baseline_route=greedy_route
    )

    best_route, generations_run = optimizer.evolve(
        max_generations=CONFIG['evolution']['generations'],
        patience=CONFIG['evolution'].get('patience', 100)
    )
    ga_dist, _, ga_collected = sim.simulate_route(best_route)


    end_ga = time.perf_counter()
    ga_times.append(end_ga - start_ga)


    print(f"GA     -> distance: {ga_dist:8.2f} | bins collected: {ga_collected:3d} | generations: {generations_run}")
    ga_total_distance += ga_dist
    ga_distances.append(ga_dist)
    
    # Track GA visits
    for bin_id in best_route[:ga_collected]:
        ga_visits[bin_id] += 1
    
    collection_numbers.append(collection_count)
    
    # Track route changes
    from src.visualization import _get_route_edges
    greedy_edges = _get_route_edges(city, greedy_route)
    ga_edges = _get_route_edges(city, best_route)
    
    removed = greedy_edges - ga_edges
    added = ga_edges - greedy_edges
    common = greedy_edges & ga_edges
    
    for edge in removed:
        all_removed_edges[edge] = all_removed_edges.get(edge, 0) + 1
    for edge in added:
        all_added_edges[edge] = all_added_edges.get(edge, 0) + 1
    for edge in common:
        all_common_edges[edge] = all_common_edges.get(edge, 0) + 1
    
    last_collection_num = collection_count
    last_greedy_route = greedy_route
    last_ga_route = best_route

    # Execute best route, updating bin states and learning rates
    best_executed_route = greedy_route if greedy_dist < ga_dist else best_route
    exec_dist, _, exec_collected = sim.execute_route(best_executed_route)
    print(f"Executed best route: distance {exec_dist:.2f}, bins {exec_collected}")

end = log_time("Simulation Total", start)

print(f"\n{'='*70}")
print("SIMULATION COMPLETE - FINAL COMPARISON")
print(f"{'='*70}")
print(f"Total collections: {collection_count}")
print(f"Total distance (Greedy): {greedy_total_distance:.2f}")
print(f"Total distance (GA):     {ga_total_distance:.2f}")
if greedy_total_distance > 0:
    improvement = (greedy_total_distance - ga_total_distance) / greedy_total_distance * 100.0
    print(f"\nGA improvement over Greedy: {improvement:.2f}%")
    print(f"Total distance saved: {greedy_total_distance - ga_total_distance:.2f}")

print(f"\n{'='*70}")
print("Timing Summary:")
print(f"Total Simulation Time: {end - start:.2f} seconds")
print(f"Greedy Times {sum(greedy_times):.2f} seconds over {collection_count} collections (avg {sum(greedy_times)/collection_count:.2f} s/collection)")
print(f"GA Times     {sum(ga_times):.2f} seconds over {collection_count} collections (avg {sum(ga_times)/collection_count:.2f} s/collection)")


# Generate visualizations
print(f"\n{'='*70}")
print("Generating visualizations...")
print(f"{'='*70}")

plot_heatmap_comparison(city, greedy_visits, ga_visits, collection_count)
plot_collection_statistics(greedy_distances, ga_distances, collection_numbers)
plot_route_comparison(city, last_greedy_route, last_ga_route, last_collection_num)
plot_aggregate_route_changes(city, all_removed_edges, all_added_edges, all_common_edges)
plot_greedy_vs_ga_times_per_day(greedy_times, ga_times, collection_numbers)

