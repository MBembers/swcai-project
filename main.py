from src.config import CONFIG
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
    width=CONFIG['city']['width'], 
    height=CONFIG['city']['height'], 
    num_points=CONFIG['city']['num_points'], 
    num_bins=CONFIG['city']['num_bins'],
    city_type=CityType.REALISTIC
)

# ===== 2. Truck (Scaled Up) =====
truck = Truck(
    truck_id=1,
    start_pos=city.depot,
    capacity=CONFIG['truck']['capacity']
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
    pop_size=CONFIG['evolution']['pop_size']
)

print("Starting Evolution...")
best_route = optimizer.evolve(generations=CONFIG['evolution']['generations'])

# ===== 4. Final Result =====
plot_simulation(city, truck, best_route, sim)
