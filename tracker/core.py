from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy.typing import NDArray 
from typing import Optional, Tuple
from geometry import SimilarityTransform2D


@dataclass
class ParamTracking:
    input_image_shape: Tuple[int, int] = (0, 0)
    pix_per_mm: float = 30
    target_pix_per_mm: float = 30
    crop_dimension_mm: Tuple[float, float] = (0, 0)
    crop_offset_y_mm: float = 0
    gamma: float = 1
    contrast: float = 1
    blur_sz_mm: float = 0
    median_filter_sz_mm: float = 0

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
        if self.crop_dimension_mm == (0, 0): 
            return (
                2 * int(self.resize*self.input_image_shape[1]//2),
                2 * int(self.resize*self.input_image_shape[0]//2)
            ) 
        else:
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
            T_input_to_global: Optional[SimilarityTransform2D]
        ) -> Optional[NDArray]:
        '''
        image: image to track, preferably background subtracted
        centroid: centroid of object to track if known 
        T_input_to_global: 3x3 coordinate transformation matrix from image coordinates to global coordinates
        return numpy structured array
        '''
        
class TrackingOverlay(ABC):

    @abstractmethod
    def overlay_global(
            self, 
            image: NDArray, 
            tracking: Optional[NDArray], 
            T_global_to_input: SimilarityTransform2D 
        ) -> Optional[NDArray]:
        pass
    
    @abstractmethod
    def overlay_cropped(self, tracking: Optional[NDArray]) -> Optional[NDArray]:
        pass

    @abstractmethod
    def overlay_processed(self, tracking: Optional[NDArray]) -> Optional[NDArray]:
        pass