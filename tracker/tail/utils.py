import math
from scipy.interpolate import splprep, splev
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

# TODO use that instead.

def tail_skeleton(
        image_crop: NDArray,
        arc_angle_deg,
        tail_length_px,
        n_tail_points,
        n_pts_arc,
        dist_swim_bladder_px,
        n_pts_interp,
        offset,
        resize,
        w
    ) -> Tuple[NDArray,NDArray]: 
        
        # track max intensity along tail
        arc_rad = math.radians(arc_angle_deg)/2
        spacing = float(tail_length_px) / n_tail_points
        start_angle = -np.pi/2
        arc = np.linspace(-arc_rad, arc_rad, n_pts_arc) + start_angle
        x = w//2 
        y = dist_swim_bladder_px
        points = [[x, y]]
        for j in range(n_tail_points):
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
                arc = np.linspace(arc[idx] - arc_rad, arc[idx] + arc_rad, n_pts_arc)
                # Add point to list
                points.append([x, y])
            except IndexError:
                points.append(points[-1])

        # interpolate
        skeleton = np.array(points).astype('float') + offset
        skeleton = skeleton / resize
        
        try:
            tck, _ = splprep(skeleton.T)
            new_points = splev(np.linspace(0,1,n_pts_interp), tck)
            skeleton_interp = np.array([new_points[0],new_points[1]])
            skeleton_interp = skeleton_interp.T
        except ValueError:
            skeleton_interp = None

        return (skeleton, skeleton_interp)