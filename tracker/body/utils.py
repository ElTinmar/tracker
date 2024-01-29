from sklearn.decomposition import PCA
from numpy.typing import NDArray
from typing import Tuple
import numpy as np

def get_orientation(coordinates: NDArray) -> Tuple[NDArray, NDArray]:
    '''
    get blob main axis using PCA
    '''

    pca = PCA()
    scores = pca.fit_transform(coordinates)
    # PCs are organized in rows, transform to columns
    principal_components = pca.components_.T
    centroid = pca.mean_

    print(coordinates)

    # resolve 180 degrees ambiguity in first PC
    if abs(max(scores[:,0])) > abs(min(scores[:,0])):
        principal_components[:,0] = - principal_components[:,0]

    # make sure the second axis always points to the same side
    if np.linalg.det(principal_components) < 0:
        principal_components[:,1] = - principal_components[:,1]
    
    return (principal_components, centroid)