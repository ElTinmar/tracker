import math
from scipy.interpolate import splprep, splev
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from numba import njit

def interpolate_skeleton(skeleton: NDArray, n_pts_interp: int) -> NDArray:
    '''
    Parametric interpolation
    skeleton: (n,2) array, where n is the number of points along the tail
    n_pts_interp: number of points for the interpolation
    '''
    
    try:
        tck, _ = splprep(skeleton.T)
        new_points = splev(np.linspace(0,1,n_pts_interp), tck)
        skeleton_interp = np.stack(new_points, axis=1) # make the array (n_pts_interp,2)
    except ValueError:
        skeleton_interp = None

    return skeleton_interp

def tail_skeleton_max(
        image: NDArray,
        arc_angle_deg: float,
        tail_length_px: int,
        n_tail_points: int,
        n_pts_arc: int,
        dist_swim_bladder_px: int,
        n_pts_interp: int,
        origin: NDArray,
        resize: float,
        w: float
    ) -> Tuple[NDArray,NDArray]: 
        '''
        Sweep a point along an arc and pick the point with max instensity.
        Carry on until you're done.
        '''
        
        # track max intensity along tail
        arc_rad = math.radians(arc_angle_deg)/2
        spacing = float(tail_length_px) / n_tail_points
        start_angle = -np.pi/2 # we are expecting to see the fish head-up and tail-down (-90 deg) 
        arc = np.linspace(-arc_rad, arc_rad, n_pts_arc) + start_angle
        x = w//2 
        y = dist_swim_bladder_px
        points = [[x, y]]
        for j in range(n_tail_points):
            try:
                # Find the x and y values of the arc centered around current x and y
                xs = x + spacing * np.cos(arc)
                ys = y - spacing * np.sin(arc)
                # Convert them to integer, because of definite pixels
                xs, ys = xs.astype(int), ys.astype(int)
                # Find the index of the minimum or maximum pixel intensity along arc
                idx = np.argmax(image[ys, xs])
                # Update new x, y points
                x = xs[idx]
                y = ys[idx]
                # Create a new arc centered around current angle
                arc = np.linspace(arc[idx] - arc_rad, arc[idx] + arc_rad, n_pts_arc)
                # Add point to list
                points.append([x, y])
            except IndexError:
                points.append(points[-1])

        skeleton = np.array(points).astype('float') + origin  #NOTE adding origin introduces a discrepancy with body coordinate system  
        skeleton = skeleton / resize
        
        # interpolate
        skeleton_interp = interpolate_skeleton(skeleton, n_pts_interp)

        return (skeleton, skeleton_interp)

@njit(cache=True)
def get_skeleton_ball(
        arc_rad, 
        n_pts_arc,
        start_angle,
        x,
        y,
        n_tail_points,
        spacing,
        image,
        ball_radius_px
    ):

    arc = np.linspace(-arc_rad, arc_rad, n_pts_arc) + start_angle
    points = [[x, y]]
    for j in range(n_tail_points-1):
        # Find the x and y values of the arc centered around current x and y
        xs = x + spacing * np.cos(arc)
        ys = y - spacing * np.sin(arc)

        # Convert them to integer, because of definite pixels
        xs, ys = xs.astype(np.int_), ys.astype(np.int_)

        # Swipe the ball along the arc and compute the sum
        max_value, x, y, a = 0, 0, 0, 0

        height, width = image.shape
        for theta, u, v in zip(arc, xs, ys):
            s = 0
            for i in range(height):
                for j in range(width):
                    if ((j - u)**2 + (i - v)**2) <= ball_radius_px**2:
                        s += image[i,j]

            if s >= max_value:
                max_value, x, y, a  = s, u, v, theta
        
        # Create a new arc centered around current angle
        arc = np.linspace(a - arc_rad, a + arc_rad, n_pts_arc)
        
        # Add point to list
        points.append([x, y])

    return points

def tail_skeleton_ball(
        image: NDArray,
        ball_radius_px: int,
        arc_angle_deg: float,
        tail_length_px: int,
        n_tail_points: int,
        n_pts_arc: int,
        dist_swim_bladder_px: int,
        n_pts_interp: int,
        origin: NDArray,
        resize: float,
        w: float
    ) -> Tuple[NDArray,NDArray]: 
        '''
        Sweep a ball along an arc and sum the pixel values inside that ball. 
        Pick the location where that sum is maximal. Carry on until you're done
        '''
        
        # track max intensity along tail
        arc_rad = math.radians(arc_angle_deg)/2
        spacing = float(tail_length_px) / n_tail_points
        start_angle = -np.pi/2 # we are expecting to see the fish head-up and tail-down (-90 deg) 
        start_x = w//2 
        start_y = dist_swim_bladder_px

        skeleton = get_skeleton_ball(
            arc_rad, 
            n_pts_arc,
            start_angle,
            start_x,
            start_y,
            n_tail_points,
            spacing,
            image,
            ball_radius_px
        )

        skeleton = np.array(skeleton).astype('float') + origin
        skeleton = skeleton / resize
        
        # interpolate
        skeleton_interp = interpolate_skeleton(skeleton, n_pts_interp)

        return (skeleton, skeleton_interp)