from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from geometry import SimilarityTransform2D


@dataclass
class Position:
    x: float
    y: float
    theta: float

class PositionPredictor(ABC):

    @abstractmethod
    def estimate(
            self, 
            tail_skeleton: np.ndarray, 
            pix_per_mm: float, 
            T_input_to_global: SimilarityTransform2D = SimilarityTransform2D.identity()
        ) -> Position:
        ...
