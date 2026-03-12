from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class SpinExchangeModel:
    """Container for symmetry-allowed exchange parameters and field settings."""

    exchanges: Dict[str, float] = field(default_factory=dict)
    magnetic_field_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    symmetry: str = "C3i"
