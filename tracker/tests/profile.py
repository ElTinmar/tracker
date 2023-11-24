from video_tools import Buffered_OpenCV_VideoReader
from image_tools import im2single, im2gray
from tracker import (
    GridAssignment, MultiFishTracker,
    AnimalTracker, AnimalTrackerParamTracking,
    BodyTracker, BodyTrackerParamTracking,
    EyesTracker, EyesTrackerParamTracking,
    TailTracker, TailTrackerParamTracking
)
from tqdm import tqdm
import numpy as np
import cProfile
import pstats
from pstats import SortKey

# background subtracted video
INPUT_VIDEO = 'toy_data/19-40-44_nobckg_static.avi'

video_reader = Buffered_OpenCV_VideoReader()
video_reader.open_file(INPUT_VIDEO)
video_reader.start()

height = video_reader.get_height()
width = video_reader.get_width()
fps = video_reader.get_fps()  
num_frames = video_reader.get_number_of_frame()

LUT = np.zeros((width, height))
assignment = GridAssignment(LUT)
accumulator = None

# tracking 
animal_tracker = AnimalTracker(
    AnimalTrackerParamTracking(
        pix_per_mm=40,
        target_pix_per_mm=7.5,
        animal_intensity=0.07,
        animal_brightness=0.0,
        animal_gamma=1.0,
        animal_contrast=1.0,
        min_animal_size_mm=1.0,
        max_animal_size_mm=30.0,
        min_animal_length_mm=1.0,
        max_animal_length_mm=12.0,
        min_animal_width_mm=0.4,
        max_animal_width_mm=2.5,
        pad_value_mm=4.0,
        blur_sz_mm=1/7.5,
        median_filter_sz_mm=1/7.5,
    )
)
body_tracker = BodyTracker(
    BodyTrackerParamTracking(
        pix_per_mm=40,
        target_pix_per_mm=7.5,
        body_intensity=0.06,
        body_brightness=0.0,
        body_gamma=1.0,
        body_contrast=1.0,
        min_body_size_mm=2.0,
        max_body_size_mm=30.0,
        min_body_length_mm=2.0,
        max_body_length_mm=6.0,
        min_body_width_mm=0.4,
        max_body_width_mm=1.2,
        blur_sz_mm=1/7.5,
        median_filter_sz_mm=1/7.5,
    )
)
eyes_tracker = EyesTracker(
    EyesTrackerParamTracking(
        pix_per_mm=40,
        target_pix_per_mm=40,
        eye_brightness=0.0,
        eye_gamma=3.0,
        eye_dyntresh_res=10,
        eye_contrast=5.0,
        eye_size_lo_mm=0.8,
        eye_size_hi_mm=10.0,
        blur_sz_mm=0.06,
        median_filter_sz_mm=0.06,
        crop_dimension_mm=(1.0,1.5),
        crop_offset_mm=-0.30
    )
)
tail_tracker = TailTracker(
    TailTrackerParamTracking(
        pix_per_mm=40,
        target_pix_per_mm=20,
        arc_angle_deg=120,
        n_tail_points=10,
        n_pts_arc=20,
        n_pts_interp=40,
        tail_length_mm=2.4,
        dist_swim_bladder_mm=0.2,
        blur_sz_mm=0.06,
        median_filter_sz_mm=0.06,
        tail_brightness=0.0,
        tail_contrast=3.0,
        tail_gamma=0.75,
        crop_dimension_mm=(3.5,3.5),
        crop_offset_tail_mm=2.25
    )
)

tracker = MultiFishTracker(            
    assignment,
    accumulator,
    animal_tracker,
    body_tracker, 
    eyes_tracker, 
    tail_tracker
)

with cProfile.Profile() as pr:

    for i in tqdm(range(num_frames)):
        (rval, frame) = video_reader.next_frame()
        if not rval:
            raise RuntimeError('VideoReader was unable to read the whole video')
        
        # convert
        frame_gray = im2single(im2gray(frame))

        # track
        tracking = tracker.track(frame_gray)

    video_reader.exit()
    video_reader.join()
    
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr).sort_stats(sortby)
ps.print_stats(50)
    
sortby = SortKey.TIME
ps = pstats.Stats(pr).sort_stats(sortby)
ps.print_stats(50)

'''
# plot cumtime vs tottime
import pandas as pd
import seaborn as sns

df = pd.DataFrame(
    pr.getstats(),
    columns=['func', 'ncalls', 'ccalls', 'tottime', 'cumtime', 'callers']
)
'''