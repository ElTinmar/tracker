from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

@dataclass
class Position:
    x: float
    y: float
    theta: float

class PositionEstimator(ABC):

    @abstractmethod
    def estimate(self, tail_skeleton: np.ndarray) -> Position:
        ...
