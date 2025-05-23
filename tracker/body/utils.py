from geometry import pca
from numpy.typing import NDArray
from typing import Optional
import numpy as np
from collections import deque

def get_best_centroid_index(
        centroids: NDArray, 
        centroid: Optional[NDArray],
    ) -> int:
    
    if centroid is None:
        return 0

    else:
        distances = np.linalg.norm(centroids - centroid, axis=1)
        closest_idx = np.argmin(distances)
    
    return closest_idx

# TODO output type incorrect (see how crop is handled) 
def get_orientation(coordinates: NDArray, heading_history: Optional[deque]) -> Optional[NDArray]: 
    '''
    get blob main axis using PCA
    '''

    # if only one point, or points aligned in 1D, quit
    if (coordinates.shape[0] <= 1) or np.any(np.var(coordinates, axis=0) == 0):
        return None

    # PCs are organized in rows, transform to columns
    body_axes, scores = pca(coordinates)

    # resolve 180 degrees ambiguity in first PC
    if abs(max(scores[:,0])) > abs(min(scores[:,0])):
        body_axes[:,0] = - body_axes[:,0]

    # prevent spurious flips using history
    if heading_history is not None:
        
        original_heading = body_axes[:,0].copy()

        if heading_history:
            past_heading = np.mean(heading_history, axis=0)
            if np.dot(past_heading, body_axes[:,0]) < 0:
                body_axes[:,0] = - body_axes[:,0]
        
        # appending anadulterated heading vectors to avoid 
        # getting stuck in the wrong orientation.
        heading_history.append(original_heading)

    # make sure the second axis always points to the same side
    if np.linalg.det(body_axes) < 0:
        body_axes[:,1] = - body_axes[:,1]
    
    return body_axes
