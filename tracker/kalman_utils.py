from enum import Enum
import numpy as np

class MotionModel(Enum):
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"
    CONSTANT_JERK = "constant_jerk"

def make_model_F(dt: float, model: MotionModel, ndim: int = 2) -> np.ndarray:
    """
    Generate a Kalman filter state transition matrix F.

    Parameters:
        dt (float): Time step.
        model (MotionModel): The motion model.
        ndim (int): Number of state variables.

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

    # Repeat for ndim dimensions using block diagonal
    F = np.block([
        [block if i == j else np.zeros_like(block) for j in range(ndim)]
        for i in range(ndim)
    ])
    return F
