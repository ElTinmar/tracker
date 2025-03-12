from sklearn.decomposition import PCA
from numpy.typing import NDArray
from typing import Tuple, List, Optional
import numpy as np
from skimage.measure._regionprops import RegionProperties

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
def get_orientation(coordinates: NDArray) -> Optional[NDArray]: 
    '''
    get blob main axis using PCA
    '''

    # if only one point, or points aligned in 1D, quit
    if (coordinates.shape[0] <= 1) or np.any(np.var(coordinates, axis=0) == 0):
        return None
    
    pca = PCA()
    scores = pca.fit_transform(coordinates)

    # PCs are organized in rows, transform to columns
    body_axes = pca.components_.T

    # resolve 180 degrees ambiguity in first PC
    if abs(max(scores[:,0])) > abs(min(scores[:,0])):
        body_axes[:,0] = - body_axes[:,0]

    # make sure the second axis always points to the same side
    if np.linalg.det(body_axes) < 0:
        body_axes[:,1] = - body_axes[:,1]
    
    return body_axes