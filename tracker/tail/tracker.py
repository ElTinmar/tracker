import math
from scipy.interpolate import splprep, splev
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from image_tools import enhance, im2uint8
from .core import TailTracker, TailTracking

class TailTracker_CPU(TailTracker):

    def track(
            self,
            image: NDArray, 
            centroid: Optional[NDArray]
        ) -> Optional[TailTracking]:

        if (image is None) or (image.size == 0) or (centroid is None):
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

        # crop image
        w, h = self.tracking_param.crop_dimension_px
        offset = np.array((-w//2, -h//2+self.tracking_param.crop_offset_tail_px), dtype=np.int32)
        left, bottom = (centroid * self.tracking_param.resize).astype(np.int32) + offset 
        right, top = left+w, bottom+h 

        image_crop = image[bottom:top, left:right]
        if image_crop.size == 0:
            return None

        # tune image contrast and gamma
        image_crop = enhance(
            image_crop,
            self.tracking_param.tail_contrast,
            self.tracking_param.tail_gamma,
            self.tracking_param.tail_brightness,
            self.tracking_param.blur_sz_px,
            self.tracking_param.median_filter_sz_px
        )

        # track max intensity along tail
        arc_rad = math.radians(self.tracking_param.arc_angle_deg)/2
        spacing = float(self.tracking_param.tail_length_px) / self.tracking_param.n_tail_points
        start_angle = -np.pi/2
        arc = np.linspace(-arc_rad, arc_rad, self.tracking_param.n_pts_arc) + start_angle
        x = w//2 
        y = self.tracking_param.dist_swim_bladder_px
        points = [[x, y]]
        for j in range(self.tracking_param.n_tail_points):
            try:
                # Find the x and y values of the arc centred around current x and y
                xs = x + spacing * np.cos(arc)
                ys = y - spacing * np.sin(arc)
                # Convert them to integer, because of definite pixels
                xs, ys = xs.astype(int), ys.astype(int)
                # Find the index of the minimum or maximum pixel intensity along arc
                idx = np.argmax(image_crop[ys, xs])
                # Update new x, y points
                x = xs[idx]
                y = ys[idx]
                # Create a new 180 arc centered around current angle
                arc = np.linspace(arc[idx] - arc_rad, arc[idx] + arc_rad, self.tracking_param.n_pts_arc)
                # Add point to list
                points.append([x, y])
            except IndexError:
                points.append(points[-1])

        # interpolate
        skeleton = np.array(points).astype('float') + offset
        skeleton = skeleton / self.tracking_param.resize
        
        try:
            tck, _ = splprep(skeleton.T)
            new_points = splev(np.linspace(0,1,self.tracking_param.n_pts_interp), tck)
            skeleton_interp = np.array([new_points[0],new_points[1]])
            skeleton_interp = skeleton_interp.T
        except ValueError:
            skeleton_interp = None

        res = TailTracking(
            centroid = centroid,
            offset = offset,
            skeleton = skeleton,
            skeleton_interp = skeleton_interp,
            image = im2uint8(image_crop)
        )    

        return res
