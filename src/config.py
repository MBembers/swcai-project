# src/config.py
import yaml
import os

CONFIG = None  # will be set by main.py

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "data", "config.yaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

