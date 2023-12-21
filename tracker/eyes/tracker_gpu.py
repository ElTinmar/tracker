import cv2
import numpy as np
from typing import Tuple, Optional
from image_tools import bwareafilter_props_GPU, bwareafilter_GPU, enhance_GPU, im2uint8, GpuMat_to_cupy_array, cupy_array_to_GpuMat
from .tracker import get_eye_prop, assign_features
import cupy as cp
from cupy.typing import NDArray as CuNDArray
from .core import EyesTracker, EyesTracking

def find_eyes_and_swimbladder_GPU(
        image: CuNDArray, 
        eye_dyntresh_res: int, 
        eye_size_lo_px: float, 
        eye_size_hi_px: float
    ) -> Tuple:
    
    # OPTIM this is slow
    thresholds = np.linspace(1/eye_dyntresh_res,1,eye_dyntresh_res)
    found_eyes_and_sb = False
    for t in thresholds:
        mask = 1.0*(image >= t)
        props = bwareafilter_props_GPU(
            mask, 
            min_size = eye_size_lo_px, 
            max_size = eye_size_hi_px
        )
        if len(props) == 3:
            found_eyes_and_sb = True
            mask = bwareafilter_GPU(
                mask, 
                min_size = eye_size_lo_px, 
                max_size = eye_size_hi_px
            )
            break

    return (found_eyes_and_sb, props, mask)

class EyesTracker_GPU(EyesTracker):

    def track(
            self,
            image: CuNDArray, 
            centroid: Optional[CuNDArray],
        ) -> Optional[EyesTracking]:

        if (image is None) or (image.size == 0) or (centroid is None):
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

        left_eye = None
        right_eye = None
        new_heading = None

        # crop image
        w, h = self.tracking_param.crop_dimension_px
        offset = cp.array((-w//2, -h//2+self.tracking_param.crop_offset_px), dtype=cp.int32)
        left, bottom = (centroid * self.tracking_param.resize).astype(cp.int32) + offset 
        right, top = left+w, bottom+h 

        image_crop = image[bottom:top, left:right]
        if image_crop.size == 0:
            return None

        # tune image contrast and gamma
        image_crop = enhance_GPU(
            image_crop,
            self.tracking_param.eye_contrast,
            self.tracking_param.eye_gamma,
            self.tracking_param.eye_brightness,
            self.tracking_param.blur_sz_px,
            self.tracking_param.median_filter_sz_px
        )

        # sweep threshold to obtain 3 connected component within size range (include SB)
        found_eyes_and_sb, props, mask = find_eyes_and_swimbladder_GPU(
            image_crop, 
            self.tracking_param.eye_dyntresh_res, 
            self.tracking_param.eye_size_lo_px, 
            self.tracking_param.eye_size_hi_px
        )
        
        offset_cpu = offset.get()
        if found_eyes_and_sb: 
            # identify left eye, right eye and swimbladder
            blob_centroids = np.array([blob.centroid for blob in props])
            sb_idx, left_idx, right_idx = assign_features(blob_centroids)

            # compute eye orientation
            left_eye = get_eye_prop(props[left_idx], offset_cpu, self.tracking_param.resize)
            right_eye = get_eye_prop(props[right_idx], offset_cpu, self.tracking_param.resize)
            #new_heading = (props[left_idx].centroid + props[right_idx].centroid)/2 - props[sb_idx].centroid
            #new_heading = new_heading / np.linalg.norm(new_heading)

        res = EyesTracking(
            centroid = centroid.get(),
            offset = offset.get(),
            left_eye = left_eye,
            right_eye = right_eye,
            mask = im2uint8(mask.get()),
            image = im2uint8(image_crop.get())
        )

        return res