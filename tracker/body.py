from image_tools import (
    bwareafilter_props, bwareafilter_props_GPU, 
    enhance, enhance_GPU, 
    im2rgb, im2uint8, 
    GpuMat_to_cupy_array, cupy_array_to_GpuMat
)
from geometry import to_homogeneous, from_homogeneous
from sklearn.decomposition import PCA
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
import cv2
from typing import Optional
from .core import Tracker, TrackingOverlay

try:
    import cupy as cp
    from cupy.typing import NDArray as CuNDArray
    from cuml.decomposition import PCA as PCA_GPU # GPU equivalent of sklearn
except:
    print('No GPU available, cupy not imported')

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

    def to_dict(self):
        res = {}
        res['pix_per_mm'] = self.pix_per_mm
        res['target_pix_per_mm'] = self.target_pix_per_mm
        res['body_intensity'] = self.body_intensity
        res['body_brightness'] = self.body_brightness
        res['body_gamma'] = self.body_gamma
        res['body_contrast'] = self.body_contrast
        res['blur_sz_mm'] = self.blur_sz_mm
        res['median_filter_sz_mm'] = self.median_filter_sz_mm
        res['min_body_size_mm'] = self.min_body_size_mm
        res['max_body_size_mm'] = self.max_body_size_mm
        res['min_body_length_mm'] = self.min_body_length_mm
        res['max_body_length_mm'] = self.max_body_length_mm
        res['min_body_width_mm'] = self.min_body_width_mm
        res['max_body_width_mm'] = self.max_body_width_mm
        return res
    
@dataclass
class BodyTrackerParamOverlay:
    pix_per_mm: float = 40.0
    heading_len_mm: float = 1.5
    heading_color: tuple = (0,128,255)
    thickness: int = 2

    def mm2px(self, val_mm):
        val_px = int(val_mm * self.pix_per_mm) 
        return val_px

    @property
    def heading_len_px(self):
        return self.mm2px(self.heading_len_mm)
        
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

def get_orientation(coordinates: NDArray) -> Tuple[NDArray, NDArray]:
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

def get_orientation_GPU(coordinates: CuNDArray) -> Tuple[CuNDArray, CuNDArray]:
    '''
    get blob main axis using PCA
    '''

    pca = PCA_GPU()
    scores = pca.fit_transform(coordinates)
    # PCs are organized in rows, transform to columns
    principal_components = pca.components_.T
    centroid = pca.mean_

    # resolve 180 degrees ambiguity in first PC
    if abs(max(scores[:,0])) > abs(min(scores[:,0])):
        principal_components[:,0] = - principal_components[:,0]

    # make sure the second axis always points to the same side
    if cp.linalg.det(principal_components) < 0:
        principal_components[:,1] = - principal_components[:,1]
    
    return (principal_components, centroid)

class BodyTracker(Tracker):

    def __init__(
            self, 
            tracking_param: BodyTrackerParamTracking, 
        ) -> None:

        self.tracking_param = tracking_param
        
    def track(
            self,
            image: NDArray, 
            centroid: Optional[NDArray] = None
        ) -> BodyTracking:
        '''
        centroid: centroid of the fish to track if it's already known.
        Useful when tracking multiple fish to discriminate between nearby blobs
        '''

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
            self.tracking_param.body_contrast,
            self.tracking_param.body_gamma,
            self.tracking_param.body_brightness,
            self.tracking_param.blur_sz_px,
            self.tracking_param.median_filter_sz_px
        )

        mask = (image >= self.tracking_param.body_intensity)
        props = bwareafilter_props(
            mask, 
            min_size = self.tracking_param.min_body_size_px,
            max_size = self.tracking_param.max_body_size_px, 
            min_length = self.tracking_param.min_body_length_px,
            max_length = self.tracking_param.max_body_length_px,
            min_width = self.tracking_param.min_body_width_px,
            max_width = self.tracking_param.max_body_width_px
        )
        
        if props == []:

            res = BodyTracking(
                heading = None,
                centroid = None,
                angle_rad = None,
                mask = im2uint8(mask),
                image = im2uint8(image)
            )
            return res
        
        else:
            if centroid is not None:
            # in case of multiple tracking, there may be other blobs
                closest_coords = None
                min_dist = None
                for blob in props:
                    row, col = blob.centroid
                    fish_centroid = np.array([col, row])
                    fish_coords = np.fliplr(blob.coords)
                    dist = np.linalg.norm(fish_centroid/self.tracking_param.resize - centroid)
                    if (min_dist is None) or (dist < min_dist): 
                        closest_coords = fish_coords
                        min_dist = dist

                (principal_components, centroid_coords) = get_orientation(closest_coords)
            else:
                fish_coords = np.fliplr(props[0].coords)
                (principal_components, centroid_coords) = get_orientation(fish_coords)

            res = BodyTracking(
                heading = principal_components,
                centroid = centroid_coords / self.tracking_param.resize,
                angle_rad = np.arctan2(principal_components[1,1], principal_components[0,1]),
                mask = im2uint8(mask),
                image = im2uint8(image)
            )
            return res

class BodyOverlay(TrackingOverlay):

    def __init__(
            self, 
            overlay_param: BodyTrackerParamOverlay
        ) -> None:

        self.overlay_param = overlay_param

    def overlay(
            self,
            image: NDArray, 
            tracking: Optional[BodyTracking], 
            transformation_matrix: NDArray 
        ) -> Optional[NDArray]:

        '''
        Coordinate system: origin = fish bounding box top left coordinates
        '''

        if (tracking is not None) and (tracking.centroid is not None):

            overlay = im2rgb(im2uint8(image))
            
            src = tracking.centroid
            heading = self.overlay_param.heading_len_px * tracking.heading[:,0]
            dst = src + heading

            # compute transformation
            pts = np.vstack((src, dst))
            pts_ = from_homogeneous((transformation_matrix @ to_homogeneous(pts).T).T)

            # heading
            overlay = cv2.line(
                overlay,
                pts_[0].astype(np.int32),
                pts_[1].astype(np.int32),
                self.overlay_param.heading_color,
                self.overlay_param.thickness
            )

            # show heading direction with a circle (easier than arrowhead)
            overlay = cv2.circle(
                overlay,
                pts_[1].astype(np.int32),
                2,
                self.overlay_param.heading_color,
                self.overlay_param.thickness
            )
        
            return overlay

class BodyTrackerGPU(Tracker):

    def track(
            self,
            image: CuNDArray, 
            centroid: Optional[CuNDArray] = None
        ) -> Optional[BodyTracking]:

        if (image is None) or (image.size == 0):
            return None

        image_gpumat = cupy_array_to_GpuMat(image)

        if self.tracking_param.resize != 1:
            image_gpumat = cv2.resize(
                image_gpumat, 
                None, 
                None,
                self.tracking_param.resize,
                self.tracking_param.resize,
                cv2.INTER_NEAREST
            )

        image = GpuMat_to_cupy_array(image_gpumat)

        # tune image contrast and gamma
        image = enhance_GPU(
            image,
            self.tracking_param.body_contrast,
            self.tracking_param.body_gamma,
            self.tracking_param.body_brightness,
            self.tracking_param.blur_sz_px,
            self.tracking_param.median_filter_sz_px
        )

        mask = (image >= self.tracking_param.body_intensity)
        props = bwareafilter_props_GPU(
            mask, 
            min_size = self.tracking_param.min_body_size_px,
            max_size = self.tracking_param.max_body_size_px, 
            min_length = self.tracking_param.min_body_length_px,
            max_length = self.tracking_param.max_body_length_px,
            min_width = self.tracking_param.min_body_width_px,
            max_width = self.tracking_param.max_body_width_px
        )
        
        if props == []:

            res = BodyTracking(
                heading = None,
                centroid = None,
                angle_rad = None,
                mask = im2uint8(mask.get()),
                image = im2uint8(image.get())
            )
            return res
        
        else:
            if centroid is not None:
            # in case of multiple tracking, there may be other blobs
                closest_coords = None
                min_dist = None
                for blob in props:
                    row, col = blob.centroid
                    fish_centroid = cp.array([col, row])
                    fish_coords = cp.fliplr(blob.coords)
                    dist = cp.linalg.norm(fish_centroid/self.tracking_param.resize - centroid)
                    if (min_dist is None) or (dist < min_dist): 
                        closest_coords = fish_coords
                        min_dist = dist

                (principal_components, centroid_coords) = get_orientation_GPU(closest_coords)
            else:
                fish_coords = cp.fliplr(props[0].coords)
                (principal_components, centroid_coords) = get_orientation_GPU(fish_coords)

            res = BodyTracking(
                heading = principal_components.get(),
                centroid = centroid_coords.get() / self.tracking_param.resize,
                angle_rad = cp.arctan2(principal_components[1,1], principal_components[0,1]),
                mask = im2uint8(mask.get()),
                image = im2uint8(image.get())
            )
            return res