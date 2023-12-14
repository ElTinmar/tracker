from dataclasses import dataclass
import math
from scipy.interpolate import splprep, splev
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from image_tools import (
    enhance, enhance_GPU, 
    im2uint8, im2rgb, 
    GpuMat_to_cupy_array, cupy_array_to_GpuMat
)
from geometry import to_homogeneous, from_homogeneous
from tracker import Tracker, TrackingOverlay

try:
    import cupy as cp
    from cupy.typing import NDArray as CuNDArray
except:
    print('No GPU available, cupy not imported')

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

    def to_dict(self):
        res = {}
        res['pix_per_mm'] = self.pix_per_mm
        res['target_pix_per_mm'] = self.target_pix_per_mm
        res['tail_contrast'] = self.tail_contrast
        res['tail_gamma'] = self.tail_gamma
        res['tail_brightness'] = self.tail_brightness
        res['arc_angle_deg'] = self.arc_angle_deg
        res['n_tail_points'] = self.n_tail_points
        res['n_pts_arc'] = self.n_pts_arc
        res['n_pts_interp'] = self.n_pts_interp
        res['tail_length_mm'] = self.tail_length_mm
        res['dist_swim_bladder_mm'] = self.dist_swim_bladder_mm
        res['blur_sz_mm'] = self.blur_sz_mm
        res['median_filter_sz_mm'] = self.median_filter_sz_mm
        res['crop_dimension_mm'] = self.crop_dimension_mm
        res['crop_offset_tail_mm'] = self.crop_offset_tail_mm
        return res
    

@dataclass
class TailTrackerParamOverlay:
    pix_per_mm: float = 40
    color_tail: tuple = (255, 128, 128)
    thickness: int = 2

@dataclass
class TailTracking:
    centroid: NDArray
    offset: NDArray # position of centroid in cropped image
    skeleton: NDArray
    skeleton_interp: NDArray
    image: NDArray

    def to_csv(self):
        '''export data as csv'''
        pass

class TailTracker(Tracker):

    def __init__(
            self, 
            tracking_param: TailTrackerParamTracking, 
        ) -> None:

        self.tracking_param = tracking_param

class TailOverlay(TrackingOverlay):

    def __init__(
            self, 
            overlay_param: TailTrackerParamOverlay
        ) -> None:

        self.overlay_param = overlay_param

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

class TailOverlay_opencv(TailOverlay):

    def overlay(
            self,
            image: NDArray, 
            tracking: Optional[TailTracking], 
            transformation_matrix: NDArray
        ) -> Optional[NDArray]:

        '''
        Coordinate system: origin = fish centroid, rotation = fish heading
        '''
                
        if tracking is not None:         

            overlay = im2rgb(im2uint8(image))       
            
            if tracking.skeleton_interp is not None:
                
                transformed_coord = from_homogeneous((transformation_matrix @ to_homogeneous(tracking.skeleton_interp).T).T)
                tail_segments = zip(transformed_coord[:-1,], transformed_coord[1:,])
                for pt1, pt2 in tail_segments:
                    overlay = cv2.line(
                        overlay,
                        pt1.astype(np.int32),
                        pt2.astype(np.int32),
                        self.overlay_param.color_tail,
                        self.overlay_param.thickness
                    )
            
            return overlay

class TailTracker_GPU(TailTracker):

    def track(
            self,
            image: CuNDArray, 
            centroid: Optional[NDArray]
        ) -> Optional[TailTracking]:

        if (image is None) or (image.size == 0) or (centroid is None):
            return None
        
        image_gpumat = cupy_array_to_GpuMat(image)

        if self.tracking_param.resize != 1:
            image = cv2.resize(
                image, 
                None, 
                None,
                self.tracking_param.resize,
                self.tracking_param.resize,
                cv2.INTER_NEAREST
            )

        image = GpuMat_to_cupy_array(image_gpumat)

        # crop image
        w, h = self.tracking_param.crop_dimension_px
        offset = np.array((-w//2, -h//2+self.tracking_param.crop_offset_tail_px), dtype=np.int32)
        left, bottom = (centroid * self.tracking_param.resize).astype(np.int32) + offset 
        right, top = left+w, bottom+h 

        image_crop = image[bottom:top, left:right]
        if image_crop.size == 0:
            return None

        # tune image contrast and gamma
        image_crop = enhance_GPU(
            image_crop,
            self.tracking_param.tail_contrast,
            self.tracking_param.tail_gamma,
            self.tracking_param.tail_brightness,
            self.tracking_param.blur_sz_px,
            self.tracking_param.median_filter_sz_px
        )

        # TODO maybe make that a function and try numba 
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
            image = im2uint8(image_crop.get())
        )    

        return res