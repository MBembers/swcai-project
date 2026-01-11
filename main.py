from src.config import CONFIG
from src.city import City, CityType, DistributionType
from src.agents import Truck
from src.simulation import Simulation
from src.evolution import GeneticOptimizer
from src.visualization import plot_simulation
from src.expert_rules import Action

# ===== 1. Setup City (Scaled Up) =====
# Increased num_points to 1200 to accommodate 300 bins comfortably
print("Generating City Graph...")
city = City(
    width=CONFIG['city']['width'], 
    height=CONFIG['city']['height'], 
    num_points=CONFIG['city']['num_points'], 
    num_bins=CONFIG['city']['num_bins'],
    city_type=CityType.REALISTIC,
  distribution_type=DistributionType.UNIFORM
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

for day in range(1, total_days + 1):
    sim.refill_bins(days=1)

    if day % interval != 0:
        continue

    collection_count += 1
    print(f"\n{'='*70}")
    print(f"Day {day}: Collection Run #{collection_count}")
    print(f"{'='*70}")
    
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

    # Optimize with GA starting from greedy
    optimizer = GeneticOptimizer(
        active_bin_ids,
        sim.get_fitness,
        pop_size=CONFIG['evolution']['pop_size'],
        baseline_route=greedy_route
    )

    best_route = optimizer.evolve(generations=CONFIG['evolution']['generations'])
    ga_dist, _, ga_collected = sim.simulate_route(best_route)
    print(f"GA     -> distance: {ga_dist:8.2f} | bins collected: {ga_collected:3d}")
    ga_total_distance += ga_dist

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

