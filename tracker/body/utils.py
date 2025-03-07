from sklearn.decomposition import PCA
from numpy.typing import NDArray
from typing import Tuple, List, Optional
import numpy as np
from skimage.measure._regionprops import RegionProperties

def get_blob_coordinates(
        centroid: Optional[NDArray], 
        props: List[RegionProperties],
        resize: float
    ) -> NDArray:
    
    if centroid is None:
        # skimage RegionProperties store coordinates as (row, col), we want (x,y)
        track_coords = np.fliplr(props[0].coords)
    else:
        # in case of multiple tracking, there may be other blobs
        track_coords = None
        min_dist = None
        for blob in props:
            row, col = blob.centroid
            fish_centroid = np.array([col, row])
            fish_coords = np.fliplr(blob.coords)
            dist = np.linalg.norm(fish_centroid/resize - centroid)
            if (min_dist is None) or (dist < min_dist): 
                track_coords = fish_coords
                min_dist = dist        
    
    return track_coords

# TODO output type incorrect (see how crop is handled) 
def get_orientation(coordinates: NDArray) -> Optional[Tuple[NDArray, NDArray]]: 
    '''
    get blob main axis using PCA
    '''

    # if only one point, or points aligned in 1D, quit
    if (coordinates.shape[0] <= 1) or np.any(np.var(coordinates, axis=0) == 0):
        return (None, None)
    
    pca = PCA()
    scores = pca.fit_transform(coordinates)

    # PCs are organized in rows, transform to columns
    principal_components = pca.components_.T
    centroid = pca.mean_

    # resolve 180 degrees ambiguity in first PC
    if abs(max(scores[:,0])) > abs(min(scores[:,0])):
        principal_components[:,0] = - principal_components[:,0]

    # make sure the second axis always points to the same side
    if np.linalg.det(principal_components) < 0:
        principal_components[:,1] = - principal_components[:,1]
    
    return (principal_components, centroid)