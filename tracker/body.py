from image_tools import bwareafilter_props, enhance, im2uint8
from sklearn.decomposition import PCA
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
import cv2
from typing import Optional
        
@dataclass
class BodyTrackerParamTracking:
    pix_per_mm: float = 40.0
    target_pix_per_mm: float = 20.0
    body_intensity: float = 0.1
    body_brightness: float = 0.2
    body_gamma: float = 1.0
    body_contrast: float = 1.0
    blur_sz_mm: float = 0.05
    median_filter_sz_mm: float = 0.15
    min_body_size_mm: float = 10.0
    max_body_size_mm: float = 100.0
    min_body_length_mm: float = 2.0
    max_body_length_mm: float = 6.0
    min_body_width_mm: float = 1.0
    max_body_width_mm: float = 3.0

    def mm2px(self, val_mm):
        return int(val_mm * self.target_pix_per_mm) 

    @property
    def resize(self):
        return self.target_pix_per_mm/self.pix_per_mm
    
    @property
    def min_body_size_px(self):
        return self.mm2px(self.min_body_size_mm)
    
    @property
    def max_body_size_px(self):
        return self.mm2px(self.max_body_size_mm) 
        
    @property
    def min_body_length_px(self):
        return self.mm2px(self.min_body_length_mm)
    
    @property
    def max_body_length_px(self):
        return self.mm2px(self.max_body_length_mm)

    @property
    def min_body_width_px(self):
        return self.mm2px(self.min_body_width_mm)
    
    @property
    def max_body_width_px(self):
        return self.mm2px(self.max_body_width_mm)

    @property
    def blur_sz_px(self):
        return self.mm2px(self.blur_sz_mm)
    
    @property
    def median_filter_sz_px(self):
        return self.mm2px(self.median_filter_sz_mm)
    
@dataclass 
class BodyTracking:
    heading: NDArray # 2x2 matrix, column 1 = fish heading, column 2 = fish right direction
    centroid: NDArray # 1x2 vector. (x,y) coordinates of the fish centroid ~ swim bladder location
    angle_rad: float
    mask: NDArray
    image: NDArray  

    def to_csv(self):
        '''
        export data to csv
        '''
        pass    

def get_orientation(coordinates: NDArray) -> NDArray:
    '''
    get blob main axis using PCA
    '''

    pca = PCA()
    scores = pca.fit_transform(coordinates)
    # PCs are organized in rows, transform to columns
    principal_components = pca.components_.T
    centroid = pca.mean_

    # resolve 180 degrees ambiguity in first PC
    if abs(max(scores[:,0])) > abs(min(scores[:,0])):
        principal_components[:,0] = - principal_components[:,0]

    # make sure the second axis always points to the same side
    if np.linalg.det(principal_components) < 0:
        principal_components[:,1] = - principal_components[:,1]
    
    return (principal_components, centroid)

def track(
        image: NDArray, 
        param: BodyTrackerParamTracking, 
        coord_centroid: Optional[NDArray] = None
    ) -> BodyTracking:
    '''
    coord_centroid: centroid of the fish to track if it's already known.
    Useful when tracking multiple fish to discriminate between nearby blobs
    '''

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
        param.body_contrast,
        param.body_gamma,
        param.body_brightness,
        param.blur_sz_px,
        param.median_filter_sz_px
    )

    mask = (image >= param.body_intensity)
    props = bwareafilter_props(
        mask, 
        min_size = param.min_body_size_px,
        max_size = param.max_body_size_px, 
        min_length = param.min_body_length_px,
        max_length = param.max_body_length_px,
        min_width = param.min_body_width_px,
        max_width = param.max_body_width_px
    )
    
    if props == []:
        return None
    else:
        if coord_centroid is not None:
        # in case of multiple tracking, there may be other blobs
            closest_coords = None
            min_dist = None
            for blob in props:
                row, col = blob.centroid
                fish_centroid = np.array([col, row])
                fish_coords = np.fliplr(blob.coords)
                dist = np.linalg.norm(fish_centroid/param.resize - coord_centroid)
                if (min_dist is None) or (dist < min_dist): 
                    closest_coords = fish_coords
                    min_dist = dist

            (principal_components, centroid) = get_orientation(closest_coords)
        else:
            fish_coords = np.fliplr(props[0].coords)
            (principal_components, centroid) = get_orientation(fish_coords)

        res = BodyTracking(
            heading = principal_components,
            centroid = centroid / param.resize,
            angle_rad = np.arctan2(principal_components[1,1], principal_components[0,1]),
            mask = im2uint8(mask),
            image = im2uint8(image)
        )
        return res
