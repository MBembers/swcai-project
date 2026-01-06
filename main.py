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
    width=200, 
    height=200, 
    num_points=1200, 
    num_bins=300, # Factor of 10x
    city_type=CityType.REALISTIC,
    distribution_type=DistributionType.UNIFORM
)

# ===== 2. Truck (Scaled Up) =====
truck = Truck(
    truck_id=1,
    start_pos=city.depot,
    capacity=6000.0 # Factor of 20x
)

sim = Simulation(city, truck)

# ===== 3. Optimization =====
# Filter bins: Only tell the GA about bins that Expert Rules say we should collect.
# This drastically reduces the search space for the GA.
print("Pre-filtering bins via Expert Rules...")
active_bin_ids = []
for b in city.bins:
    decision = sim.expert.evaluate(b)
    if decision != Action.SKIP:
        active_bin_ids.append(b.id)

print(f"Optimizing route for {len(active_bin_ids)} active bins (out of {len(city.bins)} total)...")

# ===== 3a. Baseline: Greedy Nearest-Neighbor on active bins =====
# Use City.generate_greedy_route() and filter to the active set for a fair comparison
greedy_full = city.generate_greedy_route()
active_set = set(active_bin_ids)
greedy_route = [bid for bid in greedy_full if bid in active_set]

# Evaluate greedy route
greedy_dist, greedy_penalty, greedy_collected = sim.simulate_route(greedy_route)
print(f"Greedy route -> distance: {greedy_dist:.2f}, trips/penalty proxy: {greedy_penalty:.2f}, bins: {greedy_collected}")

# ===== 3b. Genetic Algorithm Optimization =====  
# Start with greedy route in population 

optimizer = GeneticOptimizer(
    active_bin_ids,
    sim.get_fitness,
    pop_size=80, # Slightly larger population for larger search space
    baseline_route=greedy_route
)

print("Starting Evolution...")
# Run for more generations to handle the complexity
best_route = optimizer.evolve(generations=2000)

# ===== 4. Compare and Visualize =====
# Evaluate GA route
ga_dist, ga_penalty, ga_collected = sim.simulate_route(best_route)
print(f"Genetic route -> distance: {ga_dist:.2f}, trips/penalty proxy: {ga_penalty:.2f}, bins: {ga_collected}")

# Simple comparison summary
if greedy_dist > 0:
    improvement = (greedy_dist - ga_dist) / greedy_dist * 100.0
    print(f"GA vs Greedy: distance change = {greedy_dist - ga_dist:.2f} ({improvement:.2f}% relative)")
else:
    print("Greedy distance is zero; relative comparison not applicable.")

# Visualize both routes sequentially
print("Plotting Greedy route...")
# plot_simulation(city, truck, greedy_route, sim)
print("Plotting Genetic route...")
# plot_simulation(city, truck, best_route, sim)
