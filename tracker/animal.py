from image_tools import bwareafilter_centroids, enhance, im2uint8
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
import cv2

@dataclass
class AnimalTrackerParamTracking:
    pix_per_mm: float = 40.0
    target_pix_per_mm: float = 10.0
    animal_intensity: float = 0.1
    animal_brightness: float = 0.2
    animal_gamma: float = 1.0
    animal_contrast: float = 1.0
    blur_sz_mm: float = 0.05
    median_filter_sz_mm: float = 0.15
    min_animal_size_mm: float = 10.0
    max_animal_size_mm: float = 100.0
    min_animal_length_mm: float = 2.0
    max_animal_length_mm: float = 6.0
    min_animal_width_mm: float = 1.0
    max_animal_width_mm: float = 3.0
    pad_value_mm: float = 3.0

    def mm2px(self, val_mm):
        val_px = int(val_mm * self.target_pix_per_mm) 
        return val_px

    @property
    def resize(self):
        return self.target_pix_per_mm/self.pix_per_mm
    
    @property
    def min_animal_size_px(self):
        return self.mm2px(self.min_animal_size_mm)
    
    @property
    def max_animal_size_px(self):
        return self.mm2px(self.max_animal_size_mm) 
        
    @property
    def min_animal_length_px(self):
        return self.mm2px(self.min_animal_length_mm)
    
    @property
    def max_animal_length_px(self):
        return self.mm2px(self.max_animal_length_mm)

    @property
    def min_animal_width_px(self):
        return self.mm2px(self.min_animal_width_mm)
    
    @property
    def max_animal_width_px(self):
        return self.mm2px(self.max_animal_width_mm)
    
    @property
    def pad_value_px(self):
        return self.mm2px(self.pad_value_mm)

    @property
    def blur_sz_px(self):
        return self.mm2px(self.blur_sz_mm)
    
    @property
    def median_filter_sz_px(self):
        return self.mm2px(self.median_filter_sz_mm)

@dataclass
class AnimalTracking:
    centroids: NDArray # nx2 vector. (x,y) coordinates of the n fish centroid ~ swim bladder location
    bounding_boxes: NDArray
    bb_centroids: NDArray
    mask: NDArray
    image: NDArray

    def to_csv(self):
        '''
        export data to csv
        '''
        pass    

def track(image: NDArray, param: AnimalTrackerParamTracking) -> AnimalTracking:

    if (image is None) or (image.size == 0):
        return None
    
    if param.resize != 1:
        image = cv2.resize(
            image, 
            None, 
            None,
            param.resize,
            param.resize,
            cv2.INTER_NEAREST
        )

    # tune image contrast and gamma
    image = enhance(
        image,
        param.animal_contrast,
        param.animal_gamma,
        param.animal_brightness,
        param.blur_sz_px,
        param.median_filter_sz_px
    )

    height, width = image.shape
    mask = (image >= param.animal_intensity)
    centroids = bwareafilter_centroids(
        mask, 
        min_size = param.min_animal_size_px,
        max_size = param.max_animal_size_px, 
        min_length = param.min_animal_length_px,
        max_length = param.max_animal_length_px,
        min_width = param.min_animal_width_px,
        max_width = param.max_animal_width_px
    )

    bboxes = np.zeros((centroids.shape[0],4), dtype=int)
    bb_centroids = np.zeros((centroids.shape[0],2), dtype=float)
    for idx, (x,y) in enumerate(centroids):
        left = max(int(x - param.pad_value_px), 0) 
        bottom = max(int(y - param.pad_value_px), 0) 
        right = min(int(x + param.pad_value_px), width)
        top = min(int(y + param.pad_value_px), height)
        bboxes[idx,:] = [left,bottom,right,top]
        bb_centroids[idx,:] = [x-left, y-bottom] 

    res = AnimalTracking(
        centroids = centroids/param.resize,
        bounding_boxes = bboxes/param.resize,
        bb_centroids = bb_centroids/param.resize,
        mask = im2uint8(mask),
        image = im2uint8(image)
    )

    return res

