# Quick start
from main_sim import WasteCollectionSimulation

# Create simulation
sim = WasteCollectionSimulation(n_bins=30, n_trucks=1)

# Run different methods
agent_results = sim.run_agent_only_simulation()
ga_results = sim.run_ga_only_simulation()
hybrid_results = sim.run_hybrid_simulation()

# Compare methods
comparison = sim.compare_methods(n_runs=5)

city_map = sim.visualize_city_map()
