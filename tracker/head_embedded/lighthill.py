# adapted from Demarchi et al 2025, https://doi.org/10.1073/pnas.2510385122

import numpy as np
from collections import deque
from .position_predictor import PositionPredictor, Position

def get_tip_center(tail_skeleton: np.ndarray) -> np.ndarray:
    return (tail_skeleton[-2, :] + tail_skeleton[-1, :]) / 2

def get_tip_direction(tail_skeleton: np.ndarray) -> np.ndarray:
    return tail_skeleton[-1, :] - tail_skeleton[-2, :]

def cross2d(x: np.ndarray, y: np.ndarray) -> float | np.ndarray:
    """Calculates the 2D cross product (scalar or array output)."""
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

def perpendicular(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]])

def raise_to_power(signal, exponent):
    return np.sign(signal)*np.abs(signal)**exponent

def ewma(new: float, old: float, alpha: float) -> float:
    return new*alpha + old*(1.0 - alpha)

class LighthillPredictor(PositionPredictor):

    def __init__(
            self, 
            forward_gain: float = 1.0, 
            angular_gain: float = 1.0, 
            time_window_ms: int = 60,
            framerate: int = 120,
            tau: float = 0.0,
            x: float = 0.0,
            y: float = 0.0,
            theta: float = 0.0
        ):
    
        self.forward_gain = forward_gain
        self.angular_gain = angular_gain
        self.framerate = framerate
        self.alpha = 1-np.exp(-1/(framerate*tau)) if tau > 0 else 1.0

        window = int(time_window_ms/1000 * framerate)
        self.tip_center_history = deque(maxlen=3)
        self.tip_direction_history = deque(maxlen=2)
        self.force_history = deque(maxlen=window)
        self.torque_history = deque(maxlen=window)
        self.forward_speed = 0.0
        self.angular_speed = 0.0
    
        self.x = x
        self.y = y
        self.theta = theta


    def estimate(self, tail_skeleton: np.ndarray) -> Position:
        """
        tail skeleton: (N,2) numpy array 
            origin should be the center of rotation (swim-bladder)
            fish aligned with Y-axis
            units in mm
        """

        self.tip_center_history.append(get_tip_center(tail_skeleton))
        self.tip_direction_history.append(get_tip_direction(tail_skeleton))

        if len(self.tip_center_history) < 3:
            return Position(x=self.x, y=self.y, theta=self.theta)

        tip_velocity = 0.5*self.framerate*(self.tip_center_history[-1]-self.tip_center_history[0])
        tip_position = self.tip_center_history[1]
        tip_direction = self.tip_direction_history[0]

        u_perpendicular = perpendicular(tip_direction)/np.linalg.norm(tip_direction)
        u_parallel = -tip_direction/np.linalg.norm(tip_direction)
        v_perp = np.dot(tip_velocity, u_perpendicular)
        v_par = np.dot(tip_velocity, u_parallel)

        force = v_perp*(-v_par*u_perpendicular + 0.5*v_perp*u_parallel)
        torque = cross2d(tip_position, force)

        self.force_history.append(force[1])
        self.torque_history.append(torque)

        forward_speed = self.forward_gain * raise_to_power(np.nanmean(self.force_history), 2/3)
        angular_speed = self.angular_gain * np.nanmean(self.torque_history)

        self.forward_speed = ewma(forward_speed, self.forward_speed, self.alpha)
        self.angular_speed = ewma(angular_speed, self.angular_speed, self.alpha)

        dt = 1.0 / self.framerate
        forward_step_mm = self.forward_speed * dt
        angular_step_rad = self.angular_speed * dt
        
        self.theta += angular_step_rad
        self.x += forward_step_mm * np.cos(self.theta)
        self.y += forward_step_mm * np.sin(self.theta)
        
        return Position(x=self.x, y=self.y, theta=self.theta)

