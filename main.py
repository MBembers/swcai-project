from src import config  # loaded from data/config.yaml
from src.city import CityGrid
from src.agents import Truck
from src.simulation import Simulation
from src.evolution import GeneticOptimizer
from src.visualization import plot_simulation

# ===== 1. Setup City =====
city_config = config["city"]
city = CityGrid(
    width=city_config.get("width", 200),
    height=city_config.get("height", 200),
    num_bins=city_config.get("num_bins", 80)
)

# ===== 2. Truck =====
truck_config = config["truck"]
truck = Truck(
    truck_id=1,
    start_pos=city.depot,
    capacity=truck_config.get("capacity", 800.0)
)

sim = Simulation(city, truck)

# ===== 3. Evolution / Optimization =====
evo_config = config["evolution"]
all_bin_ids = [b.id for b in city.bins]

optimizer = GeneticOptimizer(
    all_bin_ids,
    sim.run_route,
    pop_size=evo_config.get("pop_size", 200)
)

print("Starting deep optimization...")
best_route = optimizer.evolve(
    generations=evo_config.get("generations", 250)
)

# ===== 4. Final Result =====
city.reset_all()
plot_simulation(city, truck, best_route, sim)
