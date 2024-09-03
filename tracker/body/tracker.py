from image_tools import  bwareafilter_props, bwareafilter_props_cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .core import BodyTracker, BodyTracking
from .utils import get_orientation
from tracker.prepare_image import prepare_image
from geometry import transform2d, Affine2DTransform


class BodyTracker_CPU(BodyTracker):
        
    def track(
            self,
            image: NDArray, 
            centroid: Optional[NDArray] = None,
            transformation_matrix: Optional[NDArray] = Affine2DTransform.identity()
        ) -> BodyTracking:
        '''
        centroid: centroid of the fish to track if it's already known.
        Useful when tracking multiple fish to discriminate between nearby blobs

        output coordinates: 
            - (0,0) = topleft corner of the bounding box
            - scale of the full-resolution image, before resizing
        '''

        if (image is None) or (image.size == 0):
            return None
        
        # pre-process image: crop/resize/tune intensity
        (origin, image_crop, image_processed) = prepare_image(
            image=image,
            source_crop_dimension_px=self.tracking_param.source_crop_dimension_px,
            target_crop_dimension_px=self.tracking_param.crop_dimension_px, 
            vertical_offset_px=0,
            centroid=centroid,
            contrast=self.tracking_param.body_contrast,
            gamma=self.tracking_param.body_gamma,
            brightness=self.tracking_param.body_brightness,
            blur_sz_px=self.tracking_param.blur_sz_px,
            median_filter_sz_px=self.tracking_param.median_filter_sz_px
        )
    
        # actual tracking starts here
        mask = (image_processed >= self.tracking_param.body_intensity)
        props = bwareafilter_props(
            mask, 
            min_size = self.tracking_param.min_body_size_px,
            max_size = self.tracking_param.max_body_size_px, 
            min_length = self.tracking_param.min_body_length_px,
            max_length = self.tracking_param.max_body_length_px,
            min_width = self.tracking_param.min_body_width_px,
            max_width = self.tracking_param.max_body_width_px
        )
        
        # if nothing is detected
        if props == []:
            res = BodyTracking(
                im_body_shape = image_processed.shape,
                im_body_fullres_shape = image_crop.shape,
                mask = mask,
                image_fullres = image_crop,
                image = image_processed
            )
            return res
        
        # something is detected
        else:
            if centroid is not None:
            # in case of multiple tracking, there may be other blobs
                track_coords = None
                min_dist = None
                for blob in props:
                    row, col = blob.centroid
                    fish_centroid = np.array([col, row])
                    fish_coords = np.fliplr(blob.coords)
                    dist = np.linalg.norm(fish_centroid/self.tracking_param.resize - centroid)
                    if (min_dist is None) or (dist < min_dist): 
                        track_coords = fish_coords
                        min_dist = dist
            else:
                track_coords = np.fliplr(props[0].coords)
            
            if track_coords.shape[0] < 2:
                res = BodyTracking(
                    im_body_shape = image_processed.shape,
                    im_body_fullres_shape = image_crop.shape,
                    mask = mask,
                    image_fullres = image_crop,
                    image = image_processed
                )
                return res
                    
            (principal_components, centroid_coords) = get_orientation(track_coords)

            centroid_ori = origin + centroid + centroid_coords / self.tracking_param.resize 

            res = BodyTracking(
                im_body_shape = image_processed.shape,
                im_body_fullres_shape = image_crop.shape,
                heading = principal_components,
                centroid = centroid_coords / self.tracking_param.resize,
                centroid_original_space = transform2d(transformation_matrix, centroid_ori),
                origin = origin,
                angle_rad = np.arctan2(principal_components[1,1], principal_components[0,1]),
                mask = mask,
                image_fullres = image_crop,
                image = image_processed
            )
            return res
