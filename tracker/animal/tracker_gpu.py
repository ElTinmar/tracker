from image_tools import bwareafilter_centroids_GPU, enhance_GPU, im2uint8, GpuMat_to_cupy_array, cupy_array_to_GpuMat
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional
from .core import AnimalTracker, AnimalTracking
from cupy.typing import NDArray as CuNDArray
from cucim.skimage import transform

class AnimalTracker_GPU(AnimalTracker):

    def track(self, image: CuNDArray, centroid: Optional[NDArray] = None) -> Optional[AnimalTracking]:

        if (image is None) or (image.size == 0):
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
        
        # tune image contrast and gamma
        image = enhance_GPU(
            image,
            self.tracking_param.animal_contrast,
            self.tracking_param.animal_gamma,
            self.tracking_param.animal_brightness,
            self.tracking_param.blur_sz_px,
            self.tracking_param.median_filter_sz_px
        )

        height, width = image.shape
        mask = (image >= self.tracking_param.animal_intensity)
        centroids_gpu = bwareafilter_centroids_GPU(
            mask, 
            min_size = self.tracking_param.min_animal_size_px,
            max_size = self.tracking_param.max_animal_size_px, 
            min_length = self.tracking_param.min_animal_length_px,
            max_length = self.tracking_param.max_animal_length_px,
            min_width = self.tracking_param.min_animal_width_px,
            max_width = self.tracking_param.max_animal_width_px
        )
        centroids = centroids_gpu.get()

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
            mask = im2uint8(mask.get()),
            image = im2uint8(image.get())
        )

        return res

class AnimalTracker_GPU_cucim(AnimalTracker):

    def track(self, image: CuNDArray, centroid: Optional[NDArray] = None) -> Optional[AnimalTracking]:

        if (image is None) or (image.size == 0):
            return None

        if self.tracking_param.resize != 1:
            image = transform.rescale(image, self.tracking_param.resize)
        
        # tune image contrast and gamma
        image = enhance_GPU(
            image,
            self.tracking_param.animal_contrast,
            self.tracking_param.animal_gamma,
            self.tracking_param.animal_brightness,
            self.tracking_param.blur_sz_px,
            self.tracking_param.median_filter_sz_px
        )

        height, width = image.shape
        mask = (image >= self.tracking_param.animal_intensity)
        centroids_gpu = bwareafilter_centroids_GPU(
            mask, 
            min_size = self.tracking_param.min_animal_size_px,
            max_size = self.tracking_param.max_animal_size_px, 
            min_length = self.tracking_param.min_animal_length_px,
            max_length = self.tracking_param.max_animal_length_px,
            min_width = self.tracking_param.min_animal_width_px,
            max_width = self.tracking_param.max_animal_width_px
        )
        centroids = centroids_gpu.get()

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
            mask = im2uint8(mask.get()),
            image = im2uint8(image.get())
        )

        return res
