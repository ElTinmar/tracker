from abc import ABC, abstractmethod
from numpy.typing import NDArray 
from typing import Any, Optional

'''
TODO Add optional output as argument to write directly into buffers:

    @abstractmethod
    def track(
            self, 
            image: NDArray,
            out: Optional[NDArray], -> write directly into buffer
            centroid: Optional[NDArray],
            transformation_matrix: Optional[NDArray]
        ) -> Optional[NDArray]:
'''

class Tracker(ABC):
    
    @abstractmethod
    def track(
            self, 
            image: NDArray,
            centroid: Optional[NDArray],
            transformation_matrix: Optional[NDArray]
        ) -> Optional[NDArray]:
        '''
        image: image to track, preferably background subtracted
        centroid: centroid of object to track if known 
        transformation_matrix: 3x3 coordinate transformation matrix from local to image coordinates
        return numpy structured array
        '''
        

class TrackingOverlay(ABC):

    @abstractmethod
    def overlay(
            self, 
            image: NDArray, 
            tracking: Optional[NDArray], 
            transformation_matrix: NDArray 
        ) -> Optional[NDArray]:
        '''
        image: image on which to overlay tracking results
        tracking: tracking results as structured array
        transformation_matrix: 3x3 coordinate transformation matrix from local to image coordinates
        return image
        '''
