
from pathlib import Path
import yaml

# ===== Paths =====
# Assumes `data/` is sibling of `src/`
DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG_PATH = DATA_DIR / "config.yaml"
FUZZY_PATH = DATA_DIR / "fuzzy_rules.yaml"

# ===== Load YAML files =====
try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Cannot find {CONFIG_PATH}")

try:
    with open(FUZZY_PATH, "r") as f:
        fuzzy_rules = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Cannot find {FUZZY_PATH}")

# ===== Optional: expose shortcuts =====
__all__ = ["config", "fuzzy_rules", "DATA_DIR"]
