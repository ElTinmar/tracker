from typing import Optional, Protocol
from numpy.typing import NDArray, ArrayLike
from dataclasses import dataclass
from tracker.core import Tracker, TrackingOverlay
import numpy as np

class Assignment(Protocol):

    def update(self):
        ...
    
    def get_ID(self):
        ...

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
    centroid_color_BGR: tuple = (128, 255, 128)
    bbox_color_BGR: tuple = (255, 255, 255) 
    centroid_thickness: int = -1
    bbox_thickness: int = 2
    id_str_color_BGR: tuple = (255, 255, 255)

    def mm2px(self, val_mm):
        return int(val_mm * self.pix_per_mm) 

    @property
    def radius_px(self):
        return self.mm2px(self.radius_mm)
    
class AnimalTracking:
    def __init__(
            self,
            im_animals_shape: tuple,
            max_num_animals: int,
            mask: NDArray,
            image: NDArray,
            identities: Optional[NDArray] = None,
            indices: Optional[NDArray] = None,
            centroids: Optional[NDArray] = None,
            bounding_boxes: Optional[NDArray] = None,
            padding: Optional[NDArray] = None,
            bb_centroids: Optional[NDArray] = None
        ) -> None:
        
        self.im_animals_shape = im_animals_shape
        self.max_num_animals = max_num_animals
        self.identities = identities
        self.indices = indices
        self.centroids = centroids # nx2 vector. (x,y) coordinates of the n fish centroid ~ swim bladder location
        self.bounding_boxes = bounding_boxes 
        self.padding = padding 
        self.bb_centroids = bb_centroids
        self.mask = mask
        self.image = image

    def to_csv(self):
        '''
        export data to csv
        '''
        pass    

    def to_numpy(self) -> NDArray:
        '''serialize to fixed-size structured numpy array'''

        dt = np.dtype([
            ('empty', bool, (1,)),
            ('max_num_animals', int, (1,)),
            ('identities', int, (self.max_num_animals, 1)),
            ('indices', int, (self.max_num_animals, 1)),
            ('centroids', np.float32, (self.max_num_animals, 2)),
            ('bounding_boxes', int, (self.max_num_animals, 4)),
            ('padding', int, (self.max_num_animals, 4)),
            ('bb_centroids', np.float32, (self.max_num_animals, 2)),
            ('mask', np.bool_, self.im_animals_shape),
            ('image', np.float32, self.im_animals_shape),
        ])
        
        arr = np.array(
            (
                self.identities is None,
                self.max_num_animals,
                np.zeros((self.max_num_animals, 1), int) if self.identities is None else self.identities,
                np.zeros((self.max_num_animals, 1), int) if self.indices is None else self.indices, 
                np.zeros((self.max_num_animals, 2), np.float32) if self.centroids is None else self.centroids, 
                np.zeros((self.max_num_animals, 4), int) if self.bounding_boxes is None else self.bounding_boxes,
                np.zeros((self.max_num_animals, 4), int) if self.padding is None else self.padding, 
                np.zeros((self.max_num_animals, 2), np.float32) if self.bb_centroids is None else self.bb_centroids, 
                self.mask, 
                self.image
            ), 
            dtype=dt
        )
        return arr
    
    @classmethod
    def from_numpy(cls, array):
        instance = cls(
            im_animals_shape = array['image'].shape,
            max_num_animals = array['max_num_animals'],
            mask = array['mask'],
            image = array['image'],
            identities = None if array['empty'] else array['identities'],
            indices = None if array['empty'] else array['indices'],
            centroids = None if array['empty'] else array['centroids'],
            bounding_boxes = None if array['empty'] else array['bounding_boxes'],
            padding = None if array['empty'] else array['padding'],
            bb_centroids = None if array['empty'] else array['bb_centroids'],
        )
        return instance
    
class AnimalTracker(Tracker):

    def __init__(
            self, 
            assignment: Assignment,
            tracking_param: AnimalTrackerParamTracking, 
        ):

        self.tracking_param = tracking_param
        self.assignment = assignment


class AnimalOverlay(TrackingOverlay):

    def __init__(
            self, 
            overlay_param: AnimalTrackerParamOverlay
        ):

        self.overlay_param = overlay_param
