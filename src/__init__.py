
from pathlib import Path
import yaml

# ===== Paths =====
# Assumes `data/` is sibling of `src/`
DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG_PATH = DATA_DIR / "config_no_rules.yaml"

# ===== Load YAML files =====
try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Cannot find {CONFIG_PATH}")


# ===== Optional: expose shortcuts =====
__all__ = ["config", "DATA_DIR"]
