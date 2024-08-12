from abc import ABC, abstractmethod
from numpy.typing import NDArray 
from typing import Any, Optional

# I know that this could be functions, I just find it easier to deal 
# with objects rather than Callable

class Tracker(ABC):
    
    @abstractmethod
    def track(
            self, 
            image: NDArray,
            centroid: Optional[NDArray],
            transformation_matrix: Optional[NDArray]
        ) -> Optional[Any]:
        '''
        image: image to track, preferably background subtracted
        centroid: centroid of object to track if known 
        '''
        

class TrackingOverlay(ABC):

    @abstractmethod
    def overlay(
            self, 
            image: NDArray, 
            tracking: Optional[Any], 
            transformation_matrix: NDArray 
        ) -> Optional[NDArray]:
        '''
        image: image on which to overlay tracking results
        tracking: tracking results
        transformation_matrix: 3x3 coordinate transformation matrix from local to image coordinates
        '''
