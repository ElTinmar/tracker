import numpy as np
from typing import Optional
from numpy.typing import NDArray
from image_tools import imrotate
from .core import MultiFishTracker, MultiFishTracking
import cv2

class MultiFishTracker_CPU(MultiFishTracker):

    def get_kwargs(self, image: NDArray, animals, body, eyes, tail) -> dict:

        kwargs = {
            'max_num_animals': self.max_num_animals,
            'animals': animals,
            'body': body,
            'eyes': eyes,
            'tail': tail
        } 

        if self.export_fullres_image:
            kwargs['image_exported'] = True
            kwargs['downsample_fullres_export'] = self.downsample_fullres_export
            if self.downsample_fullres_export != 1:
                h, w = image.shape
                dsize = (round(w*self.downsample_fullres_export), round(h*self.downsample_fullres_export))
                image = cv2.resize(
                    image,
                    dsize,
                    interpolation=cv2.INTER_NEAREST
                )
            kwargs['image'] = image

        if self.body is not None:
            body_shape = self.body.tracking_param.crop_dimension_px[::-1]
            body_fullres_shape = self.body.tracking_param.source_crop_dimension_px[::-1]
            kwargs['body_tracked'] = True
            kwargs['im_body_shape'] = body_shape
            kwargs['im_body_fullres_shape'] = body_fullres_shape      

        if self.eyes is not None:
            eyes_shape = self.eyes.tracking_param.crop_dimension_px[::-1]
            eyes_fullres_shape = self.eyes.tracking_param.source_crop_dimension_px[::-1]
            kwargs['eyes_tracked'] = True
            kwargs['im_eyes_shape'] = eyes_shape
            kwargs['im_eyes_fullres_shape'] = eyes_fullres_shape

        if self.tail is not None:
            tail_shape = self.tail.tracking_param.crop_dimension_px[::-1]
            tail_fullres_shape = self.tail.tracking_param.source_crop_dimension_px[::-1]
            kwargs['tail_tracked'] = True
            kwargs['num_tail_pts'] = self.tail.tracking_param.n_tail_points
            kwargs['num_tail_interp_pts'] = self.tail.tracking_param.n_pts_interp
            kwargs['im_tail_shape'] = tail_shape
            kwargs['im_tail_fullres_shape'] = tail_fullres_shape

        return kwargs

    def track(self, image: NDArray, centroid: Optional[NDArray] = None) -> Optional[MultiFishTracking]:

        if (image is None) or (image.size == 0):
            return None

        # get animal centroids (only crude location is necessary)
        animals = self.animal.track(image)

        body = {} 
        eyes = {}
        tail = {}

        # loop over detected animals to get body, eyes and tail tracking
        if animals.identities is not None:
            
            for id in animals.identities:

                eyes[id] = None
                tail[id] = None
                body[id] = None

                if self.body is not None:

                    # get more precise centroid and orientation of the animals
                    body[id] = self.body.track(image, centroid=animals.centroids[id,:])

                    # if body was found, track eyes and tail
                    if (body[id] is not None) and (body[id].centroid is not None):
                        
                        # rotate the animal so that it's vertical head up
                        image_rot, centroid_rot = imrotate(
                            body[id].image_fullres, 
                            body[id].centroid[0], body[id].centroid[1], 
                            np.rad2deg(body[id].angle_rad)
                        )

                        # track eyes
                        if self.eyes is not None:
                            eyes[id] = self.eyes.track(image_rot, centroid=centroid_rot)

                        # track tail
                        if self.tail is not None:
                            tail[id] = self.tail.track(image_rot, centroid=centroid_rot)

        # save tracking results and return
        kwargs = self.get_kwargs(image, animals, body, eyes, tail)
        res = MultiFishTracking(**kwargs)

        # compute additional features based on tracking
        if self.accumulator is not None:
            self.accumulator.update(res)

        return res 
    