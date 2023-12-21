from image_tools import bwareafilter_props_GPU,  enhance_GPU, im2uint8, GpuMat_to_cupy_array, cupy_array_to_GpuMat
import cv2
from typing import Optional, Tuple
from .core import BodyTracker, BodyTracking
import cupy as cp
from cupy.typing import NDArray as CuNDArray
from cuml.decomposition import PCA as PCA_GPU # GPU equivalent of sklearn

def get_orientation_GPU(coordinates: CuNDArray) -> Tuple[CuNDArray, CuNDArray]:
    '''
    get blob main axis using PCA
    '''

    pca = PCA_GPU(n_components=2)
    scores = pca.fit_transform(coordinates.astype(cp.float32))
    # PCs are organized in rows, transform to columns
    principal_components = pca.components_.T
    centroid = pca.mean_

    # resolve 180 degrees ambiguity in first PC
    if abs(cp.max(scores[:,0])) > abs(cp.min(scores[:,0])):
        principal_components[:,0] = - principal_components[:,0]

    # make sure the second axis always points to the same side
    if cp.linalg.det(principal_components) < 0:
        principal_components[:,1] = - principal_components[:,1]
    
    return (principal_components, centroid)

class BodyTracker_GPU(BodyTracker):

    def track(
            self,
            image: CuNDArray, 
            centroid: Optional[CuNDArray] = None
        ) -> Optional[BodyTracking]:

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
                angle_rad = cp.arctan2(principal_components[1,1], principal_components[0,1]).get(),
                mask = im2uint8(mask.get()),
                image = im2uint8(image.get())
            )
            return res