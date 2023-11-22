from abc import ABC, abstractclassmethod
from numpy.typing import NDArray 
from typing import Any, Optional

class Tracker(ABC):
    
    @abstractclassmethod
    def track(
        self, 
        image: NDArray,
        centroid: Optional[None]
        ) -> Optional[Any]:
        '''
        image: image to track, preferably background subtracted
        centroid: centroid of object to track if known 
        '''
        

    @abstractclassmethod
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
