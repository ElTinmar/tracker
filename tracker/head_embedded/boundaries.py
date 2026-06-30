from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Boundary(ABC):

    @abstractmethod
    def enforce(self, x: float, y: float, theta: float) -> Tuple[float, float, float]:
        pass


class NoBoundary(Boundary):

    def enforce(self, x: float, y: float, theta: float) -> Tuple[float, float, float]:
        return x, y, theta


class ClampingBoundary(Boundary):

    def __init__(self, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float]):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

    def enforce(self, x: float, y: float, theta: float) -> Tuple[float, float, float]:
        final_x = np.clip(x, self.x_bounds[0], self.x_bounds[1])
        final_y = np.clip(y, self.y_bounds[0], self.y_bounds[1])
        return final_x, final_y, theta

    
class CircularClampingBoundary(Boundary):
    """Keeps the fish inside a circular arena. 
    If it hits the wall, it slides along the perimeter."""

    def __init__(self, radius: float, center: Tuple[float, float] = (0.0, 0.0)):
        self.radius = radius
        self.cx = center[0]
        self.cy = center[1]

    def enforce(self, x: float, y: float, theta: float) -> Tuple[float, float, float]:
        dx = x - self.cx
        dy = y - self.cy
        distance = np.hypot(dx, dy)

        if distance > self.radius:
            x = self.cx + (dx / distance) * self.radius
            y = self.cy + (dy / distance) * self.radius

        return x, y, theta


class WrapAroundBoundary(Boundary):
    """Teleports the position to the opposite boundary edge (toroidal space)."""

    def __init__(self, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float]):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.width = x_bounds[1] - x_bounds[0]
        self.height = y_bounds[1] - y_bounds[0]

    def enforce(self, x: float, y: float, theta: float) -> Tuple[float, float, float]:
        final_x = self.x_bounds[0] + (x - self.x_bounds[0]) % self.width
        final_y = self.y_bounds[0] + (y - self.y_bounds[0]) % self.height
        return final_x, final_y, theta
    
