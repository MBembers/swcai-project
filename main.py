from src.city import City, CityType
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
    city_type=CityType.REALISTIC
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

optimizer = GeneticOptimizer(
    active_bin_ids,
    sim.get_fitness,
    pop_size=60 # Slightly larger population for larger search space
)

print("Starting Evolution...")
# Run for more generations to handle the complexity
best_route = optimizer.evolve(generations=60)

# ===== 4. Final Result =====
plot_simulation(city, truck, best_route, sim)
