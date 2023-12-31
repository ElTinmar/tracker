import cv2
from typing import Optional
from numpy.typing import NDArray
from image_tools import im2uint8, im2rgb
from geometry import Affine2DTransform
from .core import MultiFishOverlay, MultiFishTracking

class MultiFishOverlay_opencv(MultiFishOverlay):

    def overlay(
            self, 
            image: NDArray, 
            tracking: Optional[MultiFishTracking], 
            transformation_matrix: NDArray = Affine2DTransform.identity()
        ) -> NDArray:
        '''
        There are 3 different coordinate systems:
        - 1. image coordinates: the whole image, origin = image topleft
        - 2. bbox coordinates: cropped image of each animal, origin = bounding box top left coordinates 
        - 3. fish coordinates: fish egocentric coordinates, rotation = fish heading, origin = fish centroid
        '''

        if (tracking is not None) and (tracking.identities is not None):

            overlay = im2rgb(im2uint8(image))

            # loop over animals
            for idx, id in zip(tracking.indices, tracking.identities):

                if tracking.animals is not None:

                    # overlay animal bounding boxes, coord system 1.
                    overlay = self.animal.overlay(overlay, tracking.animals)
                    
                    # transformation matrix from coord system 1. to coord system 2., just a translation  
                    tx_bbox = tracking.animals.bounding_boxes[idx,0]
                    ty_bbox = tracking.animals.bounding_boxes[idx,1]
                    T_bbox_to_image = Affine2DTransform.translation(tx_bbox,ty_bbox)

                    if (self.body is not None) and (tracking.body[id] is not None) and (tracking.body[id].centroid is not None):

                        # overlay body, coord. system 2.
                        overlay = self.body.overlay(
                            overlay, 
                            tracking.body[id], # body coordinates in bbox
                            T_bbox_to_image
                        )

                        # transformation matrix from coord system 1. to coord system 3., rotation + translation
                        angle = tracking.body[id].angle_rad
                        rotation = Affine2DTransform.rotation(angle)
                        tx, ty = tracking.body[id].centroid 
                        T_fish_centroid_to_bbox = Affine2DTransform.translation(tx, ty)
                        T_egocentric_to_image = T_bbox_to_image @ T_fish_centroid_to_bbox @ rotation
                        
                        # overlay eyes, coord system 3.
                        if (self.eyes is not None) and (tracking.eyes[id] is not None):
                            overlay = self.eyes.overlay(
                                overlay, 
                                tracking.eyes[id], # egocentric eye coordinates
                                T_egocentric_to_image
                            )
                        
                        # overlay tail, coord system 3.
                        if (self.tail is not None) and (tracking.tail[id] is not None):
                            overlay = self.tail.overlay(
                                overlay, 
                                tracking.tail[id], # egocentric tail coordinates 
                                T_egocentric_to_image
                            )

                # show ID, coord. system 1.
                cv2.putText(overlay, str(id), (int(tx_bbox), int(ty_bbox)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv2.LINE_AA)
            
            return overlay
