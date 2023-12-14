from image_tools import  bwareafilter_props, enhance, im2uint8
from sklearn.decomposition import PCA
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional, Tuple
from .core import BodyTracker, BodyTracking

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

class BodyTracker_CPU(BodyTracker):
        
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
