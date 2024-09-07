from video_tools import InMemory_OpenCV_VideoReader
from image_tools import im2single, im2gray
from tracker import (
    GridAssignment,
    MultiFishTracker_CPU, MultiFishOverlay_opencv, MultiFishTrackerParamTracking, MultiFishTrackerParamOverlay,
    AnimalTracker_CPU, AnimalOverlay_opencv, AnimalTrackerParamTracking, AnimalTrackerParamOverlay,
    BodyTracker_CPU, BodyOverlay_opencv, BodyTrackerParamTracking, BodyTrackerParamOverlay,
    EyesTracker_CPU, EyesOverlay_opencv, EyesTrackerParamTracking, EyesTrackerParamOverlay,
    TailTracker_CPU, TailOverlay_opencv, TailTrackerParamTracking, TailTrackerParamOverlay
)
from tqdm import tqdm
import numpy as np
import cv2

# background subtracted video
VIDEOS = [
    ('toy_data/multi_freelyswimming_1800x1800px_nobckg.avi', 40),
    ('toy_data/single_freelyswimming_504x500px_nobckg.avi', 40),
    ('toy_data/single_headembedded_544x380px_noparam_nobckg.avi', 100),
    ('toy_data/single_headembedded_544x380px_param_nobckg.avi', 100)
]
# background subtracted video
INPUT_VIDEO, PIX_PER_MM = VIDEOS[1]

video_reader = InMemory_OpenCV_VideoReader()
video_reader.open_file(
    filename = INPUT_VIDEO, 
    memsize_bytes = 4e9, 
    safe = False, 
    single_precision = True, 
    grayscale = True
)

height = video_reader.get_height()
width = video_reader.get_width()
fps = video_reader.get_fps()  
num_frames = video_reader.get_number_of_frame()

LUT = np.zeros((height, width))
assignment = GridAssignment(LUT)

# tracking 
animal_tracker = AnimalTracker_CPU(
    assignment=assignment,
    tracking_param=AnimalTrackerParamTracking(
        pix_per_mm=PIX_PER_MM,
        target_pix_per_mm=7.5,
        animal_intensity=0.07,
        animal_brightness=0.0,
        animal_gamma=1.0,
        animal_contrast=1.0,
        min_animal_size_mm=1.0,
        max_animal_size_mm=30.0,
        min_animal_length_mm=0,
        max_animal_length_mm=0,
        min_animal_width_mm=0,
        max_animal_width_mm=0,
        blur_sz_mm=1/7.5,
        median_filter_sz_mm=0,
        downsample_fullres=0.25,
        num_animals=1,
        source_image_shape=(height, width)
    )
)
body_tracker = BodyTracker_CPU(
    BodyTrackerParamTracking(
        pix_per_mm=PIX_PER_MM,
        target_pix_per_mm=10,
        body_intensity=0.20,
        body_brightness=0.0,
        body_gamma=1.0,
        body_contrast=3.0,
        min_body_size_mm=0,
        max_body_size_mm=30.0,
        min_body_length_mm=0,
        max_body_length_mm=0,
        min_body_width_mm=0,
        max_body_width_mm=0,
        blur_sz_mm=1/7.5,
        median_filter_sz_mm=0,
    )
)
eyes_tracker = EyesTracker_CPU(
    EyesTrackerParamTracking(
        pix_per_mm=PIX_PER_MM,
        target_pix_per_mm=40,
        eye_brightness=0.0,
        eye_gamma=3.0,
        eye_dyntresh_res=10,
        eye_contrast=5.0,
        eye_size_lo_mm=0.8,
        eye_size_hi_mm=10.0,
        blur_sz_mm=0.06,
        median_filter_sz_mm=0,
        crop_dimension_mm=(1.0,1.5),
        crop_offset_mm=-0.75
    )
)
tail_tracker = TailTracker_CPU(
    TailTrackerParamTracking(
        pix_per_mm=PIX_PER_MM,
        target_pix_per_mm=20,
        ball_radius_mm=0.1,
        arc_angle_deg=90,
        n_tail_points=6,
        n_pts_arc=20,
        n_pts_interp=40,
        tail_length_mm=2.2,
        dist_swim_bladder_mm=0.0,
        blur_sz_mm=0.06,
        median_filter_sz_mm=0,
        tail_brightness=0.0,
        tail_contrast=3.0,
        tail_gamma=0.75,
        crop_dimension_mm=(3.5,3.5),
        crop_offset_tail_mm=1.75
    )
)

# overlay
animal_overlay = AnimalOverlay_opencv(AnimalTrackerParamOverlay())
body_overlay = BodyOverlay_opencv(BodyTrackerParamOverlay())
eyes_overlay = EyesOverlay_opencv(EyesTrackerParamOverlay())
tail_overlay = TailOverlay_opencv(TailTrackerParamOverlay())

tracker = MultiFishTracker_CPU(
    MultiFishTrackerParamTracking(
        accumulator=None,
        animal=animal_tracker,
        body=body_tracker, 
        eyes=eyes_tracker, 
        tail=tail_tracker
    )
)

overlay = MultiFishOverlay_opencv(
    MultiFishTrackerParamOverlay(
        animal_overlay,
        body_overlay,
        eyes_overlay,
        tail_overlay
    )
)

try:
    for i in tqdm(range(num_frames)):
        (rval, frame) = video_reader.next_frame()
        if not rval:
            raise RuntimeError('VideoReader was unable to read the whole video')
        
        # convert
        frame_gray = im2single(im2gray(frame))

        # track
        tracking = tracker.track(frame_gray)

        # display tracking
        oly = overlay.overlay(frame_gray, tracking)
        
        cv2.imshow('overlay',oly)
        cv2.waitKey(1)
    
finally:
    video_reader.close()
