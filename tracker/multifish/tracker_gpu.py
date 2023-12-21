import numpy as np
from typing import Optional
from numpy.typing import NDArray
from image_tools import im2uint8, imrotate_GPU, enhance_GPU, GpuMat_to_cupy_array, cupy_array_to_GpuMat
from .core import MultiFishTracker, MultiFishTracking
from cupy.typing import NDArray as CuNDArray
import cupy as cp
    
class MultiFishTracker_GPU(MultiFishTracker):

    def track(self, image: CuNDArray, centroid: Optional[NDArray] = None) -> Optional[MultiFishTracking]:

        if (image is None) or (image.size == 0):
            return None
        
        # restrain image between 0 and 1
        # image = enhance_GPU(image)

        # get animal centroids (only crude location is necessary)
        animals = self.animal.track(image)
        centroids = animals.centroids

        # if nothing was detected at that stage, stop here
        if centroids.size == 0:
            res = MultiFishTracking(
                identities =  None, 
                indices = None,
                animals = animals,
                body = None,
                eyes = None,
                tail =  None,
                image = im2uint8(image.get())
            )
            return res
        
        # assign identities to animals 
        self.assignment.update(centroids)
        identities = self.assignment.get_ID()
        to_keep = self.assignment.get_kept_centroids()        
        data = np.hstack(
            (identities[np.newaxis].T, 
             animals.bb_centroids[to_keep,:], 
             animals.bounding_boxes[to_keep,:])
        ) 

        # loop over animals
        body = {}
        eyes = {}
        tail = {}
        for (id, bb_x, bb_y, left, bottom, right, top) in data.astype(np.int64): 
            eyes[id] = None
            tail[id] = None
            body[id] = None

            # crop each animal's bounding box
            image_cropped = image[bottom:top, left:right] 
            offset = cp.array([bb_x, bb_y])
            if self.body is not None:

                # get more precise centroid and orientation of the animals
                body[id] = self.body.track(image_cropped, centroid=offset)

                if (body[id] is not None) and (body[id].centroid is not None):
                    
                    image_cropped_gpu = cupy_array_to_GpuMat(image_cropped)

                    # rotate the animal so that it's vertical head up
                    image_rot_gpu, centroid_rot = imrotate_GPU(
                        image_cropped_gpu, 
                        body[id].centroid[0], body[id].centroid[1], 
                        np.rad2deg(body[id].angle_rad)
                    )

                    image_rot = GpuMat_to_cupy_array(image_rot_gpu)

                    # track eyes 
                    if self.eyes is not None:
                        eyes[id] = self.eyes.track(image_rot, centroid=centroid_rot)

                    # track tail
                    if self.tail is not None:
                        tail[id] = self.tail.track(image_rot, centroid=centroid_rot)

                # compute additional features based on tracking
                if self.accumulator is not None:
                    self.accumulator.update(id,body[id],eyes[id],tail[id])

        # save tracking results in a dict and return
        res = MultiFishTracking(
            identities =  identities, 
            indices = to_keep,
            animals = animals,
            body = body,
            eyes = eyes,
            tail =  tail,
            image = im2uint8(image.get())
        )
        
        return res 
