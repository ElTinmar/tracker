from image_tools import bwareafilter_centroids, enhance, im2uint8
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional
from .core import AnimalTracker, AnimalTracking

class AnimalTracker_CPU(AnimalTracker):
    
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

        if centroids.size == 0:
            res = AnimalTracking(
                identities = None,
                centroids = None,
                bounding_boxes = None,
                padding = None,
                bb_centroids = None,
                mask = mask,
                image = image
            )
            return res

        bboxes = np.zeros((centroids.shape[0],4), dtype=int)
        padding = np.zeros((centroids.shape[0],4), dtype=int)
        bb_centroids = np.zeros((centroids.shape[0],2), dtype=float)
        for idx, (x,y) in enumerate(centroids):

            left = max(int(x - self.tracking_param.pad_value_px), 0)
            bottom = max(int(y - self.tracking_param.pad_value_px), 0)
            right = min(int(x + self.tracking_param.pad_value_px), width)
            top = min(int(y + self.tracking_param.pad_value_px), height)

            pad_left = -1 * min(int(x - self.tracking_param.pad_value_px), 0)
            pad_bottom = -1 * min(int(y - self.tracking_param.pad_value_px), 0)
            pad_right = -1 * min(width - int(x + self.tracking_param.pad_value_px), 0)
            pad_top = -1 * min(height - int(y + self.tracking_param.pad_value_px), 0)

            bboxes[idx,:] = [left,bottom,right,top]
            padding[idx,:] = [pad_left,pad_bottom,pad_right,pad_top]
            bb_centroids[idx,:] = [x-left, y-bottom] 

        # identity assignment
        self.assignment.update(centroids)
        identities = self.assignment.get_ID()
        to_keep = self.assignment.get_kept_centroids()   

        # resize to original
        centroids_ori = centroids[to_keep,:]/self.tracking_param.resize
        bboxes_ori = bboxes[to_keep,:]/self.tracking_param.resize
        padding_ori = padding[to_keep,:]/self.tracking_param.resize
        bb_centroids_ori = bb_centroids[to_keep,:]/self.tracking_param.resize

        res = AnimalTracking(
            identities = identities,
            indices = to_keep,
            centroids = centroids_ori,
            bounding_boxes = bboxes_ori.astype(int),
            padding = padding_ori.astype(int), # TODO make sure that stays consistent with rounding
            bb_centroids = bb_centroids_ori,
            mask = mask,
            image = image
        )
        return res
