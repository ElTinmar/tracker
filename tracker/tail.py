from dataclasses import dataclass
import math
from scipy.interpolate import splprep, splev
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from image_tools import enhance, im2uint8
from .roi_coords import get_roi_coords


@dataclass
class TailTrackerParamTracking:
    pix_per_mm: float = 40.0
    target_pix_per_mm: float = 20.0
    tail_contrast: float = 1.0,
    tail_gamma: float = 1.0,
    tail_brightness: float = 0.2,
    arc_angle_deg: float = 120.0
    n_tail_points: int = 12
    n_pts_arc: int = 20
    n_pts_interp: int = 40
    tail_length_mm: float = 2.6
    dist_swim_bladder_mm: float = 0.4
    blur_sz_mm: float = 0.10
    median_filter_sz_mm: float = 0.110
    crop_dimension_mm: Tuple[float, float] = (1.5, 1.5) 
    crop_offset_tail_mm: float = 2.25
    
    def mm2px(self, val_mm: float) -> int:
        return int(val_mm * self.target_pix_per_mm) 

    @property
    def resize(self):
        return self.target_pix_per_mm/self.pix_per_mm
       
    @property
    def tail_length_px(self):
        return self.mm2px(self.tail_length_mm)
    
    @property
    def dist_swim_bladder_px(self):
        return self.mm2px(self.dist_swim_bladder_mm)

    @property
    def blur_sz_px(self):
        return self.mm2px(self.blur_sz_mm) 

    @property
    def median_filter_sz_px(self):
        return self.mm2px(self.median_filter_sz_mm) 

    @property
    def crop_offset_tail_px(self):
        return self.mm2px(self.crop_offset_tail_mm) 
    
    @property
    def crop_dimension_px(self):
        return (
            self.mm2px(self.crop_dimension_mm[0]),
            self.mm2px(self.crop_dimension_mm[1])
        ) 
    
@dataclass
class TailTracking:
    centroid: NDArray
    skeleton: NDArray
    skeleton_interp: NDArray
    image: NDArray

    def to_csv(self):
        '''export data as csv'''
        pass


def track(
        image: NDArray, 
        param: TailTrackerParamTracking, 
        centroid: NDArray
    ) -> TailTracking:

    if (image is None) or (image.size == 0):
        return None

    if param.resize != 1:
        image = cv2.resize(
            image, 
            None, 
            None,
            param.resize,
            param.resize,
            cv2.INTER_NEAREST
        )

    # crop image
    left, bottom, w, h = get_roi_coords(
        centroid, 
        param.crop_dimension_px, 
        param.crop_offset_tail_px, 
        param.resize
    )
    right = left + w
    top = bottom + h
    image_crop = image[bottom:top, left:right]
    if image_crop.size == 0:
        return None

    # tune image contrast and gamma
    image_crop = enhance(
        image_crop,
        param.tail_contrast,
        param.tail_gamma,
        param.tail_brightness,
        param.blur_sz_px,
        param.median_filter_sz_px
    )

    # track max intensity along tail
    arc_rad = math.radians(param.arc_angle_deg)/2
    spacing = float(param.tail_length_px) / param.n_tail_points
    start_angle = -np.pi/2
    arc = np.linspace(-arc_rad, arc_rad, param.n_pts_arc) + start_angle
    x = w//2 
    y = param.dist_swim_bladder_px
    points = [[x, y]]
    for j in range(param.n_tail_points):
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
            arc = np.linspace(arc[idx] - arc_rad, arc[idx] + arc_rad, param.n_pts_arc)
            # Add point to list
            points.append([x, y])
        except IndexError:
            points.append(points[-1])

    # interpolate
    skeleton = np.array(points).astype('float')
    skeleton = skeleton / param.resize
    
    try:
        tck, _ = splprep(skeleton.T)
        new_points = splev(np.linspace(0,1,param.n_pts_interp), tck)
        skeleton_interp = np.array([new_points[0],new_points[1]])
        skeleton_interp = skeleton_interp.T
    except ValueError:
        skeleton_interp = None

    res = TailTracking(
        centroid = centroid,
        skeleton = skeleton,
        skeleton_interp = skeleton_interp,
        image = im2uint8(image_crop)
    )    

    return res
    
