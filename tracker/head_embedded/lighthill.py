# adapted from Demarchi et al 2025, https://doi.org/10.1073/pnas.2510385122

import numpy as np
from collections import deque
from geometry import SimilarityTransform2D
from .position_predictor import PositionPredictor, Position
from .boundaries import Boundary, NoBoundary  

def get_tip_center(tail_skeleton: np.ndarray) -> np.ndarray:
    return (tail_skeleton[-2, :] + tail_skeleton[-1, :]) / 2


def get_tip_direction(tail_skeleton: np.ndarray) -> np.ndarray:
    return tail_skeleton[-1, :] - tail_skeleton[-2, :]


def get_heading_direction(tail_skeleton: np.ndarray) -> np.ndarray:
    return tail_skeleton[0, :] - tail_skeleton[1, :]


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
            forward_gain: float = 0.08, # (s/mm)^(1/3)
            angular_gain: float = 0.01, # rad⋅s/mm^3
            time_window_ms: int = 30,
            framerate: int = 120,
            tau: float = 0.0,
            boundary: Boundary = None  
        ):
    
        self.forward_gain = forward_gain
        self.angular_gain = angular_gain
        self.framerate = framerate
        self.alpha = 1-np.exp(-1/(framerate*tau)) if tau > 0 else 1.0
        self.boundary = boundary or NoBoundary()

        window = int(time_window_ms/1000 * framerate)
        self.tip_center_history = deque(maxlen=3)
        self.tip_direction_history = deque(maxlen=2)
        #self.heading_direction_history = deque(maxlen=2)
        self.force_history = deque(maxlen=window)
        self.torque_history = deque(maxlen=window)

        self.forward_speed = 0.0
        self.angular_speed = 0.0
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def estimate(
            self, 
            tail_skeleton: np.ndarray, # shape (N,2)
            pix_per_mm: float,
            T_input_to_global: SimilarityTransform2D = SimilarityTransform2D.identity()
        ) -> Position:
        # tail skeleton with fish head oriented north

        def world_to_image(x_local, y_local, theta_local):
            # transform back to image space
            x = (x_local * pix_per_mm) + tail_skeleton[0,0]
            y = (-y_local * pix_per_mm) + tail_skeleton[0,1]
            theta = -theta_local

            # extra transformation to a global space if needed
            coords_global = T_input_to_global.transform_points(np.array([x, y])).squeeze()
            angle_global = T_input_to_global.transform_angles(theta)

            return coords_global[0], coords_global[1], angle_global[0] 

        # transform left-handed image space to right-handed coordinate system centered on
        # first tail point, in world coordinates (mm), forward axis y=[0, 1]
        tail_mm = tail_skeleton.copy() / pix_per_mm
        tail_mm = (tail_mm - tail_mm[0, :]) 
        tail_mm[:,1] = -tail_mm[:,1] 

        self.tip_center_history.append(get_tip_center(tail_mm))
        self.tip_direction_history.append(get_tip_direction(tail_mm))
        #self.heading_direction_history.append(get_heading_direction(tail_mm))

        # if we don't have enough data to compute central difference, return early
        if len(self.tip_center_history) < 3:
            return Position(*world_to_image(self.x,self.y,self.theta))

        tip_velocity = 0.5*self.framerate*(self.tip_center_history[-1]-self.tip_center_history[0])
        tip_position = self.tip_center_history[1]
        tip_direction = self.tip_direction_history[0]
        #heading_direction = self.heading_direction_history[0]

        #u_forward = heading_direction/np.linalg.norm(heading_direction)
        u_forward = np.array([0.0, 1.0])
        u_perpendicular = perpendicular(tip_direction)/np.linalg.norm(tip_direction)
        u_parallel = -tip_direction/np.linalg.norm(tip_direction)
        v_perp = np.dot(tip_velocity, u_perpendicular)
        v_par = np.dot(tip_velocity, u_parallel)

        force = v_perp*(-v_par*u_perpendicular + 0.5*v_perp*u_parallel) 
        torque = cross2d(tip_position, force)

        self.force_history.append(force @ u_forward)  
        self.torque_history.append(torque)

        forward_speed = self.forward_gain * raise_to_power(np.nanmean(self.force_history), 2/3)
        forward_speed = max(0, np.nan_to_num(forward_speed, nan=self.forward_speed))
        angular_speed = self.angular_gain * np.nanmean(self.torque_history)
        angular_speed = np.nan_to_num(angular_speed, nan=self.angular_speed)

        self.forward_speed = ewma(forward_speed, self.forward_speed, self.alpha)
        self.angular_speed = ewma(angular_speed, self.angular_speed, self.alpha)

        dt = 1.0 / self.framerate
        forward_step_mm = self.forward_speed * dt
        angular_step_rad = self.angular_speed * dt
        
        new_theta = self.theta + angular_step_rad 
        new_x = self.x + forward_step_mm * np.cos(self.theta) 
        new_y = self.y + forward_step_mm * np.sin(self.theta)

        self.x, self.y, self.theta = self.boundary.enforce(new_x, new_y, new_theta)

        return Position(*world_to_image(self.x,self.y,self.theta))

