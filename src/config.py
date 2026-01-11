import yaml
import os

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default to data/config.yaml relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'data', 'config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

# Load config once when module is imported
CONFIG = load_config()
