from image_tools import im2gray, filter_contours_centroids, filter_connected_comp_centroids, filter_floodfill_centroid
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional, Tuple
from .core import AnimalTracker
from tracker.prepare_image import preprocess_image, Preprocessing
from tracker.core import Resolution
from geometry import SimilarityTransform2D
from filterpy.common import kinematic_kf
from dataclasses import dataclass

@dataclass
class AnimalResolution(Resolution):
    pix_per_mm_downsampled: float = 0.0

class Tracking:
    def __init__(self, num_animals: int) -> None:
        self.num_animals: int = num_animals
        self.centroids_resized: NDArray = np.zeros((num_animals, 2), np.float32)
        self.centroids_cropped: NDArray = np.zeros((num_animals, 2), np.float32)
        self.centroids_input: NDArray = np.zeros((num_animals, 2), np.float32)
        self.centroids_global: NDArray = np.zeros((num_animals, 2), np.float32)

class AnimalTracker_CPU(AnimalTracker):

    def transform_input_centroid(
            self, 
            centroid, # centroid in global space
            T_input_to_global
        ) -> Tuple[Optional[NDArray], SimilarityTransform2D]:
        
        T_global_to_input = T_input_to_global.inv()

        if centroid is None:
            centroid_input = None
        else:
            centroid_input = T_global_to_input.transform_points(centroid).squeeze()

        return centroid_input, T_global_to_input        

    def preprocess(
        self,
        image: NDArray, 
        background_image: NDArray, 
        centroid: Optional[NDArray] = None, # centroids in input space
        ) -> Optional[Preprocessing]:
        
        return preprocess_image(image, background_image, centroid, self.tracking_param)

    def track_resized(
            self,
            preproc: Preprocessing, 
        ) -> Optional[Tuple[Tracking, NDArray]]:
        
        tracking = Tracking(num_animals=self.tracking_param.num_animals)
        mask = cv2.compare(
            preproc.image_processed, 
            self.tracking_param.intensity, # type: ignore
            cv2.CMP_GT
        ) 
        centroids_resized = filter_connected_comp_centroids(
            mask, 
            min_size = self.tracking_param.min_size_px,
            max_size = self.tracking_param.max_size_px, 
            min_length = None if self.tracking_param.min_length_px == 0 else self.tracking_param.min_length_px,
            max_length = None if self.tracking_param.max_length_px == 0 else self.tracking_param.max_length_px,
            min_width = None if self.tracking_param.min_width_px == 0 else self.tracking_param.min_width_px,
            max_width = None if self.tracking_param.max_width_px == 0 else self.tracking_param.max_width_px
        )     

        if centroids_resized.size == 0:
            return None
        
        # TODO: fix assignment for multifish to work in resized space
        tracking.centroids_resized = self.assignment.update(centroids_resized)  
        
        return tracking, mask

    def transform_coordinate_system(
            self,
            tracking: Tracking,        
            preproc: Preprocessing,
            T_input_to_global: SimilarityTransform2D,
            T_global_to_input: SimilarityTransform2D
        ) -> AnimalResolution:
        '''modifies tracking in-place with side-effect and returns resolution'''
        
        tracking.centroids_cropped = preproc.T_resized_to_cropped.transform_points(tracking.centroids_resized)
        tracking.centroids_input = preproc.T_cropped_to_input.transform_points(tracking.centroids_cropped)
        tracking.centroids_global = T_input_to_global.transform_points(tracking.centroids_input)

        resolution = AnimalResolution()
        resolution.pix_per_mm_global = self.tracking_param.pix_per_mm
        resolution.pix_per_mm_input = resolution.pix_per_mm_global * T_global_to_input.scale_factor
        resolution.pix_per_mm_cropped = resolution.pix_per_mm_input * preproc.T_input_to_cropped.scale_factor
        resolution.pix_per_mm_resized = resolution.pix_per_mm_cropped * preproc.T_cropped_to_resized.scale_factor
        resolution.pix_per_mm_downsampled = resolution.pix_per_mm_cropped * self.tracking_param.downsample_factor

        return resolution
    
    def track(
        self,
        image: NDArray, # image in input space
        background_image: Optional[NDArray] = None,
        centroid: Optional[NDArray] = None, # centroid in global space
        T_input_to_global: SimilarityTransform2D = SimilarityTransform2D.identity() # input to global space transform
    ) -> NDArray:
        
        # TODO this is bit of a hack
        self.tracking_param.crop_dimension_mm = (image.shape[1]/self.tracking_param.pix_per_mm, image.shape[0]/self.tracking_param.pix_per_mm) 
        self.tracking_param.input_image_dtype = image.dtype
        
        # only work with one channel
        image = im2gray(image)

        if background_image is None:
            background_image = np.zeros_like(image)
        background_image = im2gray(background_image)
        
        centroid_in_input, T_global_to_input = self.transform_input_centroid(
            centroid,
            T_input_to_global
        )

        preproc = self.preprocess(image, background_image, centroid_in_input) 
        if preproc is None:
            return self.tracking_param.failed
        
        # Downsample image export (a bit easier on RAM). This is used for overlay instead of image_cropped
        # NOTE: it introduces a special case, not a big fan of this
        image_downsampled = cv2.resize(
            src = preproc.image_cropped,
            dsize = self.tracking_param.downsampled_shape[::-1], # transform shape (row, col) to width, height
            interpolation = cv2.INTER_NEAREST
        )
        
        tracking_resized = self.track_resized(preproc)
        if tracking_resized is None:
            return self.tracking_param.failed

        tracking, mask = tracking_resized
        resolution = self.transform_coordinate_system(
            tracking, 
            preproc, 
            T_input_to_global, 
            T_global_to_input
        )
        
        # NOTE: for large image/many fish, creating that array might take time
        # maybe pre-allocate in tracking_param?
        res = np.array(
            (
                True,
                self.tracking_param.num_animals,
                tracking.centroids_resized,
                tracking.centroids_cropped,
                tracking.centroids_input,
                tracking.centroids_global, 
                self.tracking_param.downsample_factor,
                mask, 
                preproc.image_processed,
                image_downsampled,
                resolution.pix_per_mm_global,
                resolution.pix_per_mm_input,
                resolution.pix_per_mm_cropped,
                resolution.pix_per_mm_resized,
                resolution.pix_per_mm_downsampled
            ),
            dtype=self.tracking_param.dtype
        )
        return res


class AnimalTrackerKalman(AnimalTracker_CPU):

    def __init__(
            self, 
            fps: float, 
            model_order: int, 
            model_uncertainty: float = 0.2,
            measurement_uncertainty: float = 1.0,
            *args, 
            **kwargs
        ) -> None:

        super().__init__(*args, **kwargs)
        self.N_DIM = 2 * self.tracking_param.num_animals
        self.fps = fps
        dt = 1/fps
        self.kalman_filter = kinematic_kf(
            dim = self.N_DIM, 
            order = model_order, 
            dt = dt, 
            dim_z = self.N_DIM, 
            order_by_dim = False
        )
        self.kalman_filter.Q *= model_uncertainty
        self.kalman_filter.R *= measurement_uncertainty

    def tracking_to_measurement(self, tracking: Tracking) -> NDArray:
        
        measurement = np.zeros((self.N_DIM,1))
        measurement[0:self.N_DIM,0] = tracking.centroids_resized.flatten()
        return measurement

    def prediction_to_tracking(self, tracking: Tracking) -> None:
        '''Side effect: modify tracking in-place'''
        
        tracking.centroids_resized = self.kalman_filter.x[0:self.N_DIM,0].reshape((self.tracking_param.num_animals,2))

    def return_prediction_if_tracking_failed(
            self,
            preproc: Preprocessing,
            T_input_to_global: SimilarityTransform2D,
            T_global_to_input: SimilarityTransform2D,
        ) -> NDArray:
        
        tracking = Tracking(num_animals=self.tracking_param.num_animals)
        self.kalman_filter.predict()
        self.kalman_filter.update(None)
        self.prediction_to_tracking(tracking)

        resolution = self.transform_coordinate_system(
            tracking, 
            preproc, 
            T_input_to_global, 
            T_global_to_input
        )

        res = np.array(
            (
                True,
                self.tracking_param.num_animals,
                tracking.centroids_resized,
                tracking.centroids_cropped,
                tracking.centroids_input,
                tracking.centroids_global, 
                self.tracking_param.downsample_factor,
                np.zeros_like(preproc.image_processed), 
                preproc.image_processed,
                np.zeros(self.tracking_param.downsampled_shape, np.float32),
                resolution.pix_per_mm_global,
                resolution.pix_per_mm_input,
                resolution.pix_per_mm_cropped,
                resolution.pix_per_mm_resized,
                resolution.pix_per_mm_downsampled
            ),
            dtype=self.tracking_param.dtype
        )
        return res

    def track(
            self,
            image: NDArray, 
            background_image: Optional[NDArray] = None, 
            centroid: Optional[NDArray] = None, # centroids in global space
            T_input_to_global: SimilarityTransform2D = SimilarityTransform2D.identity()
        ) -> NDArray:

        # only work with one channel
        image = im2gray(image)
        self.tracking_param.input_image_dtype = image.dtype
        
        if background_image is None:
            background_image = np.zeros_like(image)
        background_image = im2gray(background_image)

        centroid_in_input, T_global_to_input = self.transform_input_centroid(
            centroid,
            T_input_to_global
        )

        preproc = self.preprocess(image, background_image, centroid_in_input) 
        if preproc is None:
            return self.return_prediction_if_tracking_failed(
                preproc, # TODO this is None make sure it's ok
                T_input_to_global,
                T_global_to_input,
            )
        
        image_downsampled = cv2.resize(
            src = preproc.image_cropped,
            dsize = self.tracking_param.downsampled_shape[::-1], # transform shape (row, col) to width, height
            interpolation = cv2.INTER_NEAREST
        )
        
        tracking_resized = self.track_resized(preproc)
        if tracking_resized is None:
            return self.return_prediction_if_tracking_failed(
                preproc,
                T_input_to_global,
                T_global_to_input,
            )
        
        # kalman filter
        tracking, mask = tracking_resized
        self.kalman_filter.predict()
        measurement = self.tracking_to_measurement(tracking)
        self.kalman_filter.update(measurement)
        self.prediction_to_tracking(tracking)

        resolution = self.transform_coordinate_system(
            tracking, 
            preproc, 
            T_input_to_global, 
            T_global_to_input
        )
        
        res = np.array(
            (
                True,
                self.tracking_param.num_animals,
                tracking.centroids_resized,
                tracking.centroids_cropped,
                tracking.centroids_input,
                tracking.centroids_global, 
                self.tracking_param.downsample_factor,
                mask, 
                preproc.image_processed,
                image_downsampled,
                resolution.pix_per_mm_global,
                resolution.pix_per_mm_input,
                resolution.pix_per_mm_cropped,
                resolution.pix_per_mm_resized,
                resolution.pix_per_mm_downsampled
            ),
            dtype=self.tracking_param.dtype
        )
        return res
    