from typing import Optional
from numpy.typing import NDArray, ArrayLike
from dataclasses import dataclass
from tracker.core import Tracker, TrackingOverlay
import numpy as np

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
    
    def to_dict(self):
        res = {}
        res['pix_per_mm'] = self.pix_per_mm
        res['target_pix_per_mm'] = self.target_pix_per_mm
        res['animal_intensity'] = self.animal_intensity
        res['animal_brightness'] = self.animal_brightness
        res['animal_gamma'] = self.animal_gamma
        res['animal_contrast'] = self.animal_contrast
        res['blur_sz_mm'] = self.blur_sz_mm
        res['median_filter_sz_mm'] = self.median_filter_sz_mm
        res['min_animal_size_mm'] = self.min_animal_size_mm
        res['max_animal_size_mm'] = self.max_animal_size_mm
        res['min_animal_length_mm'] = self.min_animal_length_mm
        res['max_animal_length_mm'] = self.max_animal_length_mm
        res['min_animal_width_mm'] = self.min_animal_width_mm
        res['max_animal_width_mm'] = self.max_animal_width_mm
        res['pad_value_mm'] = self.pad_value_mm
        return res

@dataclass
class AnimalTrackerParamOverlay:
    pix_per_mm: float = 40.0
    radius_mm: float = 0.1
    centroid_color: tuple = (255, 128, 128)
    bbox_color:tuple = (255, 255, 255) 
    centroid_thickness: int = -1
    bbox_thickness: int = 2

    def mm2px(self, val_mm):
        return int(val_mm * self.pix_per_mm) 

    @property
    def radius_px(self):
        return self.mm2px(self.radius_mm)
    
class AnimalTracking:
    def __init__(
            self,
            im_shape: Optional[ArrayLike],
            max_num_animals: Optional[int],
            centroids: Optional[NDArray],
            bounding_boxes: Optional[NDArray],
            bb_centroids: Optional[NDArray],
            mask: Optional[NDArray],
            image: Optional[NDArray],
        ) -> None:
        
        self.centroids = centroids if centroids is not None else np.zeros((max_num_animals,2), dtype = np.single) # nx2 vector. (x,y) coordinates of the n fish centroid ~ swim bladder location
        self.bounding_boxes = bounding_boxes  if bounding_boxes is not None else np.zeros((max_num_animals,4), dtype = np.single)
        self.bb_centroids = bb_centroids  if bb_centroids is not None else np.zeros((max_num_animals,2), dtype = np.single)
        self.mask = mask if mask is not None else np.zeros(im_shape, dtype = np.uint8)
        self.image = image if image is not None else np.zeros(im_shape, dtype = np.uint8)

    def to_csv(self):
        '''
        export data to csv
        '''
        pass    

    def to_numpy(self) -> NDArray:
        '''serialize to structured numpy array'''
        dt = np.dtype([
            ('centroid', self.centroids.dtype, self.centroid.shape),
            ('bounding_boxes',  self.bounding_boxes.dtype, self.bounding_boxes.shape),
            ('bb_centroids',  self.bb_centroids.dtype, self.bb_centroids.shape),
            ('mask',  self.mask.dtype, self.mask.shape),
            ('image',  self.image.dtype, self.image.shape),
        ])
        arr = np.array((self.centroids, self.bounding_boxes, self.bb_centroids, self.mask, self.image), dtype=dt)
        return arr
    
class AnimalTracker(Tracker):

    def __init__(
            self, 
            tracking_param: AnimalTrackerParamTracking, 
        ):

        self.tracking_param = tracking_param

class AnimalOverlay(TrackingOverlay):

    def __init__(
            self, 
            overlay_param: AnimalTrackerParamOverlay
        ):

        self.overlay_param = overlay_param
