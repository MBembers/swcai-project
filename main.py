from src.config import CONFIG
from src.city import City, CityType, DistributionType
from src.agents import Truck
from src.simulation import Simulation
from src.evolution import GeneticOptimizer
from src.visualization import plot_simulation, plot_heatmap_comparison, plot_collection_statistics, plot_route_comparison, plot_aggregate_route_changes
from src.expert_rules import Action

# ===== 1. Setup City (Scaled Up) =====
# Increased num_points to 1200 to accommodate 300 bins comfortably
print("Generating City Graph...")
city = City(
    width=CONFIG['city']['width'], 
    height=CONFIG['city']['height'], 
    num_points=CONFIG['city']['num_points'], 
    num_bins=CONFIG['city']['num_bins'],
    city_type=CityType.MANHATTAN,
  distribution_type=DistributionType.EXPONENTIAL_DECAY
)

# ===== 2. Truck (Scaled Up) =====
truck = Truck(
    truck_id=1,
    start_pos=city.depot,
    capacity=CONFIG['truck']['capacity']
)

sim = Simulation(city, truck)

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

    # Optimize with GA starting from greedy
    optimizer = GeneticOptimizer(
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

# Generate visualizations
print(f"\n{'='*70}")
print("Generating visualizations...")
print(f"{'='*70}")

plot_heatmap_comparison(city, greedy_visits, ga_visits, collection_count)
plot_collection_statistics(greedy_distances, ga_distances, collection_numbers)
plot_route_comparison(city, last_greedy_route, last_ga_route, last_collection_num)
plot_aggregate_route_changes(city, all_removed_edges, all_added_edges, all_common_edges)


