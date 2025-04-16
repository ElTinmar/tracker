from enum import Enum
import numpy as np
from filterpy.kalman import KalmanFilter

class MotionModel(Enum):
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"
    CONSTANT_JERK = "constant_jerk"

def state_transition(dt: float, model: MotionModel, dim_z) -> np.ndarray:
    """
    Generate a Kalman filter state transition matrix F.

    Parameters:
        dt (float): Time step.
        model (MotionModel): The motion model.
        dim_z (int): Number of state variables.

    Returns:
        F (np.ndarray): State transition matrix.
    """

    # Define base F for 1D motion
    if model == MotionModel.CONSTANT_VELOCITY:
        block = np.array([
            [1, dt],
            [0, 1],
        ])
    elif model == MotionModel.CONSTANT_ACCELERATION:
        block = np.array([
            [1, dt, 0.5 * dt ** 2],
            [0, 1, dt],
            [0, 0, 1],
        ])
    elif model == MotionModel.CONSTANT_JERK:
        block = np.array([
            [1, dt, 0.5 * dt**2, (1/6) * dt**3],
            [0, 1, dt, 0.5 * dt**2],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ])
    else:
        raise ValueError(f"Unknown model: {model}")

    # Repeat for dim_z dimensions using block diagonal
    F = np.block([
        [block if i == j else np.zeros_like(block) for j in range(dim_z)]
        for i in range(dim_z)
    ])
    return F

def measurement_matrix(dim_x: int, dim_z: int) -> np.ndarray:

    H = np.zeros((dim_z, dim_x))
    for i in range(dim_z):
        H[i, i * dim_x//dim_z] = 1

    return H
    
def create_kalman_filter(dt: float, model: MotionModel, dim_z) -> KalmanFilter:
    
    if model == MotionModel.CONSTANT_VELOCITY:
        dim_x = dim_z * 2

    elif model == MotionModel.CONSTANT_ACCELERATION:
        dim_x = dim_z * 3

    elif model == MotionModel.CONSTANT_JERK:
        dim_x = dim_z * 4

    else:
        raise ValueError(f"Unknown model: {model}")
    
    kalman_filter = KalmanFilter(dim_x, dim_z)
    kalman_filter.x = np.zeros((dim_x,1))
    kalman_filter.F = state_transition(dt, model, dim_z)
    kalman_filter.H = measurement_matrix(dim_x, dim_z)
    kalman_filter.P = 100 * np.eye(dim_x)
    kalman_filter.Q = np.eye(dim_x)
    kalman_filter.R = np.eye(dim_z)

    return kalman_filter
