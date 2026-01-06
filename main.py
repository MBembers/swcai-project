from src.city import CityGrid
from src.agents import Truck
from src.simulation import Simulation
from src.evolution import GeneticOptimizer
from src.visualization import plot_simulation

# 1. Setup: 30 bins
city = CityGrid(200, 200, num_bins=80)

# 2. Truck Capacity: Set to 800 (allows ~8-12 bins per trip)
truck = Truck(truck_id=1, start_pos=city.depot, capacity=800.0)
sim = Simulation(city, truck)

# 3. Evolution
all_bin_ids = [b.id for b in city.bins]
# More generations and larger population for 30 bins
optimizer = GeneticOptimizer(all_bin_ids, sim.run_route, pop_size=200)

print("Starting deep optimization...")
best_route = optimizer.evolve(generations=250)

# 4. Final Result
city.reset_all()
plot_simulation(city, truck, best_route, sim)
