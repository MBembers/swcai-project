import csv
import random
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt


class Bin:
    """Represents a waste bin with location and fill properties."""

    def __init__(self, x: int, y: int, fill: float, fill_uncertainty: float, fill_speed: float):
        """
        Initialize a Bin.

        Args:
            x: X coordinate (0-100)
            y: Y coordinate (0-100)
            fill: Current fill level (0-100)
            fill_uncertainty: Uncertainty in fill measurement (0-100)
            fill_speed: Fill speed in units per day
        """
        self.x = x
        self.y = y
        self.fill = fill
        self.fill_uncertainty = fill_uncertainty
        self.fill_speed = fill_speed

    def __repr__(self):
        return (f"Bin(x={self.x}, y={self.y}, fill={self.fill}, "
                f"fill_uncertainty={self.fill_uncertainty}, fill_speed={self.fill_speed})")

    def to_dict(self):
        """Convert bin to dictionary for CSV export."""
        return {
            'x': self.x,
            'y': self.y,
            'fill': self.fill,
            'fill_uncertainty': self.fill_uncertainty,
            'fill_speed': self.fill_speed
        }

    @staticmethod
    def from_dict(data: dict):
        """Create a Bin from a dictionary."""
        return Bin(
            x=int(data['x']),
            y=int(data['y']),
            fill=float(data['fill']),
            fill_uncertainty=float(data['fill_uncertainty']),
            fill_speed=float(data['fill_speed'])
        )


class BinsGenerator:
    """Generates, saves, and loads waste bins."""

    def __init__(self, grid_size: int = 100):
        """
        Initialize the BinsGenerator.

        Args:
            grid_size: Size of the grid (default 100x100)
        """
        self.grid_size = grid_size
        self.bins: List[Bin] = []

    def generate_bins(self, num_bins: int) -> List[Bin]:
        """
        Randomly generate bins on the grid.

        Args:
            num_bins: Number of bins to generate

        Returns:
            List of generated Bin objects
        """
        self.bins = []
        for _ in range(num_bins):
            bin_obj = Bin(
                x=random.randint(0, self.grid_size),
                y=random.randint(0, self.grid_size),
                fill=random.uniform(0, 100),
                fill_uncertainty=abs(random.gauss(0, 10)),
                fill_speed=max(10, random.gauss(30, 10))
            )
            self.bins.append(bin_obj)
        return self.bins

    def save_to_csv(self, filename: str) -> None:
        """
        Save bins to an indexed CSV file.

        Args:
            filename: Path to the CSV file
        """
        if not self.bins:
            raise ValueError(
                "No bins to save. Generate bins first using generate_bins().")

        filepath = Path("data") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['index', 'x', 'y', 'fill',
                          'fill_uncertainty', 'fill_speed']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for idx, bin_obj in enumerate(self.bins):
                row = {'index': idx}
                row.update(bin_obj.to_dict())
                writer.writerow(row)

    def load_from_csv(self, filename: str) -> List[Bin]:
        """
        Load bins from a CSV file.

        Args:
            filename: Path to the CSV file

        Returns:
            List of Bin objects loaded from file
        """
        self.bins = []
        filepath = Path("data") / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Skip the index column as it's just for reference
                bin_obj = Bin.from_dict(row)
                self.bins.append(bin_obj)

        return self.bins

    def plot_bins(self, title: str = "Waste Bins Distribution"):
        """
        Visualize bins on a scatter plot colored by fill level.

        Args:
            title: Title for the plot
        """
        if not self.bins:
            raise ValueError("No bins to plot. Generate or load bins first.")

        x_coords = [bin_obj.x for bin_obj in self.bins]
        y_coords = [bin_obj.y for bin_obj in self.bins]
        fill_levels = [bin_obj.fill for bin_obj in self.bins]

        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(x_coords, y_coords, c=fill_levels, cmap='RdYlGn_r',
                              s=100, alpha=0.6, edgecolors='black', linewidth=1)

        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.title(title, fontsize=14)
        plt.xlim(-5, self.grid_size + 5)
        plt.ylim(-5, self.grid_size + 5)
        plt.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter)
        cbar.set_label('Fill Level (%)', fontsize=12)

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate bins
    generator = BinsGenerator()
    generator.generate_bins(10)

    print("Generated bins:")
    for bin_obj in generator.bins:
        print(bin_obj)

    # Save to CSV
    csv_file = "bins.csv"
    generator.save_to_csv(csv_file)
    print(f"\nBins saved to {csv_file}")

    # Load from CSV
    new_generator = BinsGenerator()
    loaded_bins = new_generator.load_from_csv(csv_file)

    print(f"\nLoaded {len(loaded_bins)} bins from {csv_file}:")
    for bin_obj in loaded_bins:
        print(bin_obj)

    # Plot bins colored by fill level
    new_generator.plot_bins("Waste Bins - Colored by Fill Level")
