import math
from scipy.interpolate import splprep, splev
import cv2
import numpy as np
from typing import Optional
from image_tools import enhance_GPU, im2uint8, GpuMat_to_cupy_array, cupy_array_to_GpuMat
from .core import TailTracker, TailTracking
import cupy as cp
from cupy.typing import NDArray as CuNDArray
from .utils import tail_skeleton

class TailTracker_GPU(TailTracker):

    def track(
            self,
            image: CuNDArray, 
            centroid: Optional[CuNDArray]
        ) -> Optional[TailTracking]:

        if (image is None) or (image.size == 0) or (centroid is None):
            return None
        
        image_gpumat = cupy_array_to_GpuMat(image)

        if self.tracking_param.resize != 1:
            image_gpumat = cv2.cuda.resize(
                image_gpumat, 
                None, 
                None,
                self.tracking_param.resize,
                self.tracking_param.resize,
                cv2.INTER_NEAREST
            )

        image = GpuMat_to_cupy_array(image_gpumat)

        # crop image
        w, h = self.tracking_param.crop_dimension_px
        offset = cp.array((-w//2, -h//2+self.tracking_param.crop_offset_tail_px), dtype=cp.int32)
        left, bottom = (centroid * self.tracking_param.resize).astype(cp.int32) + offset 
        right, top = left+w, bottom+h 

        image_crop = image[bottom:top, left:right]
        if image_crop.size == 0:
            return None

        # tune image contrast and gamma
        image_crop = enhance_GPU(
            image_crop,
            self.tracking_param.tail_contrast,
            self.tracking_param.tail_gamma,
            self.tracking_param.tail_brightness,
            self.tracking_param.blur_sz_px,
            self.tracking_param.median_filter_sz_px
        )

        # TODO maybe make that a function and try numba 
        image_crop_cpu = image_crop.get()
        offset_cpu = offset.get()
        skeleton, skeleton_interp = tail_skeleton(
            image_crop_cpu,
            self.tracking_param.arc_angle_deg,
            self.tracking_param.tail_length_px,
            self.tracking_param.n_tail_points,
            self.tracking_param.n_pts_arc,
            self.tracking_param.dist_swim_bladder_px,
            self.tracking_param.n_pts_interp,
            offset_cpu,
            self.tracking_param.resize,
            w
        )

        res = TailTracking(
            centroid = centroid.get(),
            offset = offset_cpu,
            skeleton = skeleton,
            skeleton_interp = skeleton_interp,
            image = im2uint8(image_crop_cpu)
        )    

        return res