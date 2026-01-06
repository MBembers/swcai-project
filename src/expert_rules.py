from enum import Enum
from .city import Bin

class Action(Enum):
    COLLECT = 1
    SKIP = 2
    URGENT = 3

class ExpertRules:
    """
    Layer 1: Expert Rules
    Determines if a bin is worth visiting based on simple heuristics.
    """
    def __init__(self, min_fill_threshold=0.40, critical_threshold=0.90):
        self.min_fill = min_fill_threshold
        self.critical = critical_threshold

    def evaluate(self, bin_obj: Bin) -> Action:
        fill_ratio = bin_obj.fill_level / bin_obj.capacity

        # Rule 1: Criticality
        if fill_ratio >= self.critical:
            return Action.URGENT
        
        # Rule 2: Efficiency (Skip empty/low bins)
        if fill_ratio < self.min_fill:
            return Action.SKIP

        return Action.COLLECT
