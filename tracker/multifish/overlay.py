from typing import Optional
import numpy as np
from numpy.typing import NDArray
from image_tools import im2uint8, im2rgb
from geometry import Affine2DTransform
from .core import MultiFishOverlay

class MultiFishOverlay_opencv(MultiFishOverlay):

    def overlay_cropped(self, tracking: Optional[NDArray]) -> Optional[NDArray]:
        pass

    def overlay_processed(self, tracking: Optional[NDArray]) -> Optional[NDArray]:
        pass

    def overlay_global(
            self, 
            image: NDArray, 
            tracking: Optional[NDArray], 
            transformation_matrix: NDArray = Affine2DTransform.identity()
        ) -> NDArray:
        '''
        There are 3 different coordinate systems:
        - 1. image coordinates: the whole image, origin = image topleft
        - 2. bbox coordinates: cropped image of each animal, origin = bounding box top left coordinates 
        - 3. fish coordinates: fish egocentric coordinates, rotation = fish heading, origin = fish centroid
        '''

        if (tracking is not None):

            T_scale = Affine2DTransform.scaling(
                tracking['animals']['downsample_ratio'],
                tracking['animals']['downsample_ratio']
            ) 

            overlay = im2rgb(im2uint8(image))

            # overlay animal bounding boxes, coord system 1.
            overlay = self.overlay_param.animal.overlay_global(overlay, tracking['animals'], T_scale)         

            # loop over animals
            for idx, _ in enumerate(tracking['animals']['centroids_global']):

                if (
                        (self.overlay_param.body is not None) 
                        and ('body' in tracking.dtype.fields) 
                    ):

                    # transformation matrix from coord system 1. to coord system 2., just a translation  
                    tx, ty = tracking['animals']['centroids_global'][idx,:] - np.asarray(tracking['body'][idx]['image_cropped'].shape[::-1])//2 # dirty fix?
                    T_bbox_to_image = Affine2DTransform.translation(tx,ty)
                    
                    # overlay body, coord. system 2.
                    overlay = self.overlay_param.body.overlay_global(
                        overlay, 
                        tracking['body'][idx], # body coordinates in bbox
                        T_scale @ T_bbox_to_image
                    )

                    # transformation matrix from coord system 1. to coord system 3., rotation + translation
                    angle = tracking['body'][idx]['angle_rad']
                    rotation = Affine2DTransform.rotation(angle)
                    tx, ty = tracking['body'][idx]['centroid_resized']
                    T_fish_centroid_to_bbox = Affine2DTransform.translation(tx, ty)
                    T_egocentric_to_image = T_scale @ T_bbox_to_image @ T_fish_centroid_to_bbox @ rotation 
                    
                    # overlay eyes, coord system 3.
                    if (
                            (self.overlay_param.eyes is not None) 
                            and ('eyes' in tracking.dtype.fields)
                        ):

                        overlay = self.overlay_param.eyes.overlay_global(
                            overlay, 
                            tracking['eyes'][idx], # egocentric eye coordinates
                            T_egocentric_to_image
                        )
                    
                    # overlay tail, coord system 3.
                    if (
                            (self.overlay_param.tail is not None) 
                            and ('tail' in tracking.dtype.fields)
                        ):

                        overlay = self.overlay_param.tail.overlay_global(
                            overlay, 
                            tracking['tail'][idx], # egocentric tail coordinates 
                            T_egocentric_to_image
                        )

            return overlay
