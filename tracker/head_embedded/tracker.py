import numpy as np
from typing import Optional
from numpy.typing import NDArray
from image_tools import imrotate, im2gray
from geometry import SimilarityTransform2D
from .core import HeadEmbeddedTracker

class HeadEmbeddedTracker_CPU(HeadEmbeddedTracker):

    def track(
            self, 
            image: NDArray, 
            background_image: Optional[NDArray] = None,
            centroid: Optional[NDArray] = None,
            T_input_to_global: SimilarityTransform2D = SimilarityTransform2D.identity()
        ) -> NDArray:
        
        # only work with one channel
        image = im2gray(image)

        if background_image is None:
            background_image = np.zeros_like(image)
        background_image = im2gray(background_image)
        
        # rotate the animal so that it's vertical head up
        image_rot, centroid_rot = imrotate(
            image, 
            self.tracking_param.centroid_x, self.tracking_param.centroid_y,
            np.rad2deg(self.tracking_param.heading_angle_rad)
        )
        background_rot, _ = imrotate(
            background_image, 
            self.tracking_param.centroid_x, self.tracking_param.centroid_y,
            np.rad2deg(self.tracking_param.heading_angle_rad)
        )

        T = SimilarityTransform2D.translation(self.tracking_param.centroid_x, self.tracking_param.centroid_y)
        R = SimilarityTransform2D.rotation(self.tracking_param.heading_angle_rad)
        T0 = SimilarityTransform2D.translation(-centroid_rot[0], -centroid_rot[1])
        T_image_rot_to_global = T_input_to_global @ T @ R @ T0

        tail = self.tracking_param.tail.track(
            image_rot, 
            background_rot, 
            centroid, 
            T_image_rot_to_global
        )

        position = self.tracking_param.position_estimator.estimate(tail['skeleton_global'])
        arr = (True, position.x, position.y, position.theta, tail)
        res = np.array(
            arr,
            dtype=self.tracking_param.dtype
        )

        return res
    