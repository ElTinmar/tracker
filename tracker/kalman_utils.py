from geometry import angdiff
from numpy.typing import NDArray
from filterpy.kalman import KalmanFilter
import numpy as np

def kalman_update_wrap_angle(
        kalman_filter: KalmanFilter, 
        measurement: NDArray, 
        angle_dim: NDArray
    ) -> None:
    '''
    Tracked angles can jump suddenly by 2*pi. This update function
    computes angular difference for the innovation in [-pi,pi] and 
    wraps predicted angles in [-pi,pi].
    Update kalman filter in-place with side-effect.
    '''
    
    innovation = measurement - kalman_filter.H @ kalman_filter.x
    innovation[angle_dim, 0] = angdiff(
        measurement[angle_dim, 0], 
        (kalman_filter.H @ kalman_filter.x)[angle_dim, 0]
    )  

    # Kalman update
    S = kalman_filter.H @ kalman_filter.P @ kalman_filter.H.T + kalman_filter.R
    K = kalman_filter.P @ kalman_filter.H.T @ np.linalg.inv(S)
    kalman_filter.x += K @ innovation
    kalman_filter.P = (np.eye(kalman_filter.dim_x) - K @ kalman_filter.H) @ kalman_filter.P

    # Wrap angle state back into [-pi,pi]
    kalman_filter.x[angle_dim, 0] = np.angle(np.exp(1j * kalman_filter.x[angle_dim, 0]))