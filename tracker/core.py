from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy.typing import NDArray 
from typing import Optional, Tuple

@dataclass
class ParamTracking:
    pix_per_mm: float = 40.0
    target_pix_per_mm: float = 20.0
    crop_dimension_mm: Tuple[float, float] = (5.5, 5.5) 
    crop_offset_y_mm: float = 0.0
    intensity: float = 0.1
    gamma: float = 1.0
    contrast: float = 1.0
    blur_sz_mm: float = 0.05
    median_filter_sz_mm: float = 0.15
    do_crop: bool = True
    do_resize: bool = True
    do_enhance: bool = True

    def target_mm2px(self, val_mm: float) -> int:
        return int(val_mm * self.target_pix_per_mm) 

    def source_mm2px(self, val_mm):
        return int(val_mm * self.pix_per_mm) 
    
    @property
    def resize(self):
        return self.target_pix_per_mm/self.pix_per_mm

    @property
    def blur_sz_px(self):
        return self.target_mm2px(self.blur_sz_mm) 

    @property
    def median_filter_sz_px(self):
        return self.target_mm2px(self.median_filter_sz_mm) 

    @property
    def crop_offset_y_px(self):
        return self.target_mm2px(self.crop_offset_y_mm) 

    @property
    def resized_dimension_px(self):
        # some video codec require height, width to be divisible by 2
        return (
            2 * (self.target_mm2px(self.crop_dimension_mm[0])//2),
            2 * (self.target_mm2px(self.crop_dimension_mm[1])//2)
        ) 
    
    @property
    def crop_dimension_px(self):
        # some video codec require height, width to be divisible by 2
        return (
            2* (self.source_mm2px(self.crop_dimension_mm[0])//2),
            2* (self.source_mm2px(self.crop_dimension_mm[1])//2)
        ) 
    
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
