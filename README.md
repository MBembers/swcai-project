<a name="top"></a>
[![language](https://img.shields.io/badge/language-Python-blue)](https://learn.microsoft.com/en-us/shows/intro-to-python-development/)
[![OS](https://img.shields.io/badge/OS-linux%2C%20windows-0078D4)]()
[![CPU](https://img.shields.io/badge/CPU-x86%2C%20x64%2C%20ARM%2C%20ARM64-FF8C00)](https://docs.abblix.com/docs/technical-requirements)
[![Free](https://img.shields.io/badge/free_for_non_commercial_use-brightgreen)](#-license)

# Smart Waste-Collection Routing with Uncertain Bin Levels

## Project Overview

This project addresses the **Smart Waste-Collection Routing Problem** with uncertain bin fill levels. The aim of this project is to optimize waste collection truck routes in a simulated urban environment where:

- **Bin fill levels are uncertain**: Waste bins have stochastic fill rates with per-bin individual characteristics
- **Observations are uncertain**: Actual fill levels are observed with measurement noise

Our goal is to minimize distance traveled in order to reduce environmental impact, avoid overflow situations, and handle resource constraints if applied in real life situations.

---

## Project Architecture

The system is organized into multiple layers:

### 1. **City & Environment Layer** ([`src/city.py`](src/city.py))
- Creates a simulated urban environment with configurable road networks
- Supports different city types:
  - `REALISTIC`: Voronoi diagram-based more realistic street layouts
  - `MANHATTAN`: Manhattan grid layout
- Distributes waste bins with configurable patterns:
  - `UNIFORM`: Uniform distribution across the city
  - `EXPONENTIAL_DECAY`: Higher bin density the closer to the center (to simulate city center)
- Each bin has:
  - Individual capacity (randomly chosen from given options)
  - Individual fill rate (with random noise, filled each day of the simulation)
  - Current fill level (stochastic)

### 2. **Simulation** ([`src/simulation.py`](src/simulation.py))
- Manages temporal dynamics of bin fill levels
- Handles:
  - Bin refilling between collection cycles (with stochastic noise)
  - Observation sampling (uncertain measurements)
  - Learning/adaptation of per-bin fill rate estimates
  - Collection event tracking
- Updates bin states based on empirical observations

### 3. **Expert Rules** ([`src/expert_rules.py`](src/expert_rules.py))
- Simple heuristic rules for initial bin evaluation:
  - `SKIP`: Bin fill < min_fill_threshold (40%) -> not worth visiting
  - `COLLECT`: Bin fill in [40%, 90%) -> normal collection
  - `URGENT`: Bin fill >= critical_threshold (90%) -> must visit

### 4. **Optimization**
Multiple algorithms for route optimization:

- **Genetic Algorithm** ([`src/evolution.py`](src/evolution.py)):
  - Population-based optimization
  - Order/two-point crossover
  - Tournament selection
  - Configurable elitism and mutation rates

- **L-SHADE Optimizer** ([`src/lshade.py`](src/lshade.py)):
  - Differential Evolution variant
  - Adaptive parameter control
  - Memory-based parameter selection
  - Ranking-based permutation decoding

- **AL-SHADE Optimizer** (augmented version):
  - Enhanced L-SHADE with additional learning mechanisms

### 5. **Truck & Routing** ([`src/agents.py`](src/agents.py))
- Represents waste collection trucks with:
  - Fixed capacity (e.g., 6000 units)
  - Current load tracking
  - Route management
  - Distance calculation

### 6. **Visualization** ([`src/visualization.py`](src/visualization.py))
- Visualization outputs for analyzing the run:
  - City graph with bin and depot locations
  - Route visualization overlayed on city
  - Collection heatmaps
  - Comparative analysis charts
  - Performance metrics over time

---

## Configuration Files

Configurations are managed through YAML files in the `data/` directory.

### **Preset Configurations**

- **`config.yaml`**: Default config file
- **`config_debug.yaml`**: Debugging configuration (fewer bins, fewer generations)
- **`config_no_rules.yaml`**: Runs optimization without expert rules constraints
- **`config_LSHADE.yaml`**: Uses L-SHADE optimizer instead of genetic algorithm
- **`config_ALSHADE.yaml`**: Uses augmented L-SHADE optimizer

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running with Default Configuration
```bash
python main.py
```

### Running with Custom Configuration
```bash
python main.py config_LSHADE.yaml
```

---

## Dependencies

- **NumPy**: Numerical computations
- **SciPy**: Spatial algorithms (Voronoi diagrams)
- **PyYAML**: Configuration file parsing
- **Matplotlib**: Visualization and plotting
- **NetworkX**: Graph operations for city networks

---

## Project Outputs

### Console Output
- Configuration details and loading confirmation
- City graph generation progress
- Simulation day-by-day results
- Optimization progress (fitness improvements, generations completed)
- Final route statistics and performance metrics

### Generated Results
- **`results.txt`**: Detailed run results and performance summary
- **`plots/`**: Visualization outputs including:
  - City layout with bin/depot positions
  - Optimized collection routes
  - Heatmaps of collection patterns
  - Performance comparison charts

---
## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.