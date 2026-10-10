import numpy as np
from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class Config:

    # Lattice
    cell: str = 'honeycomb'
    cluster: str = 'A'
    n_max: List[int] = field(default_factory=lambda: [2, 3])

    # Occupation
    filling: float = 0.5
    Sz: float = 0.0

    # Hoppings and twisted boundary flux
    t1: float = 1.0
    t2: float = 0.2
    phi: float = 0.5

    # ED solver parameters
    tol: float = 0

    # Flux grid per dimension
    n_flux: int = 10

    # Disorder parameters
    n_samples: int = 1
    w0: float = 0.0
    w1: float = 0.0
    w2: float = 0.0