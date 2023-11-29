from image_tools import bwareafilter_centroids, enhance, im2uint8, im2rgb
from geometry import to_homogeneous, Affine2DTransform
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
import cv2
from typing import Optional, Any
from .core import Tracker, TrackingOverlay

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

class AnimalTracker(Tracker):

    def __init__(
            self, 
            tracking_param: AnimalTrackerParamTracking, 
        ):

        self.tracking_param = tracking_param

    def track(self, image: NDArray, centroid: Optional[NDArray] = None) -> Optional[AnimalTracking]:

        if (image is None) or (image.size == 0):
            return None
        
        if self.tracking_param.resize != 1:
            image = cv2.resize(
                image, 
                None, 
                None,
                self.tracking_param.resize,
                self.tracking_param.resize,
                cv2.INTER_NEAREST
            )
        
        # tune image contrast and gamma
        image = enhance(
            image,
            self.tracking_param.animal_contrast,
            self.tracking_param.animal_gamma,
            self.tracking_param.animal_brightness,
            self.tracking_param.blur_sz_px,
            self.tracking_param.median_filter_sz_px
        )

        height, width = image.shape
        mask = (image >= self.tracking_param.animal_intensity)
        centroids = bwareafilter_centroids(
            mask, 
            min_size = self.tracking_param.min_animal_size_px,
            max_size = self.tracking_param.max_animal_size_px, 
            min_length = self.tracking_param.min_animal_length_px,
            max_length = self.tracking_param.max_animal_length_px,
            min_width = self.tracking_param.min_animal_width_px,
            max_width = self.tracking_param.max_animal_width_px
        )

        bboxes = np.zeros((centroids.shape[0],4), dtype=int)
        bb_centroids = np.zeros((centroids.shape[0],2), dtype=float)
        for idx, (x,y) in enumerate(centroids):
            left = max(int(x - self.tracking_param.pad_value_px), 0) 
            bottom = max(int(y - self.tracking_param.pad_value_px), 0) 
            right = min(int(x + self.tracking_param.pad_value_px), width)
            top = min(int(y + self.tracking_param.pad_value_px), height)
            bboxes[idx,:] = [left,bottom,right,top]
            bb_centroids[idx,:] = [x-left, y-bottom] 

        res = AnimalTracking(
            centroids = centroids/self.tracking_param.resize,
            bounding_boxes = bboxes/self.tracking_param.resize,
            bb_centroids = bb_centroids/self.tracking_param.resize,
            mask = im2uint8(mask),
            image = im2uint8(image)
        )

        return res

class AnimalTrackerGPU(Tracker):
    
    def track(self, image: NDArray, centroid: Optional[NDArray] = None) -> Optional[AnimalTracking]:
        '''TODO'''

class AnimalOverlay(TrackingOverlay):

    def __init__(
            self, 
            overlay_param: AnimalTrackerParamOverlay
        ):

        self.overlay_param = overlay_param

    def overlay(
            self,
            image: NDArray, 
            tracking: Optional[AnimalTracking], 
            transformation_matrix: NDArray = Affine2DTransform.identity()
        ) -> Optional[NDArray]:

        if (tracking is not None) and (tracking.centroids.size > 0):

            overlay = im2rgb(im2uint8(image))

            # draw centroid
            for x,y,_ in (transformation_matrix @ to_homogeneous(tracking.centroids).T).T:
                overlay = cv2.circle(
                    overlay,
                    (int(x),int(y)), 
                    self.overlay_param.radius_px, 
                    self.overlay_param.centroid_color, 
                    self.overlay_param.centroid_thickness
                )

            # draw bounding boxes
            for (left, bottom, right, top) in tracking.bounding_boxes:
                rect = np.array([[left, top],[right, bottom]])
                bbox = (transformation_matrix @ to_homogeneous(rect).T).T
                topleft = bbox[0,:-1].astype(int)
                bottomright = bbox[1,:-1].astype(int)
                overlay = cv2.rectangle(
                    overlay, 
                    topleft,
                    bottomright, 
                    self.overlay_param.bbox_color, 
                    self.overlay_param.bbox_thickness
                )

            return overlay
        
