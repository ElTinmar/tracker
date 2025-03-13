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
from geometry import SimilarityTransform2D

DISPLAY=False

# background subtracted video
VIDEOS = [
    ('toy_data/multi_freelyswimming_1800x1800px_nobckg.avi', 40),
    ('toy_data/single_freelyswimming_504x500px_nobckg.avi', 40),
    ('toy_data/single_headembedded_544x380px_noparam_nobckg.avi', 100),
    ('toy_data/single_headembedded_544x380px_param_nobckg.avi', 100)
]
# background subtracted video
VIDEO_NUM = 0
INPUT_VIDEO, PIX_PER_MM = VIDEOS[VIDEO_NUM]

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
num_animals = 1
if VIDEO_NUM == 0:
    LUT[0:600,0:600] = 0
    LUT[0:600,600:1200] = 1
    LUT[0:600,1200:1800] = 2
    LUT[600:1200,0:600] = 3
    LUT[600:1200,600:1200] = 4
    LUT[600:1200,1200:1800] = 5
    LUT[1200:1800,0:600] = 6
    LUT[1200:1800,600:1200] = 7
    LUT[1200:1800,1200:1800] = 8
    num_animals = 9

assignment = GridAssignment(LUT, num_animals)

# tracking 
animal_tracker = AnimalTracker_CPU(
    assignment=assignment,
    tracking_param=AnimalTrackerParamTracking(
        pix_per_mm=PIX_PER_MM,
        target_pix_per_mm=5,
        intensity=0.15,
        gamma=1.0,
        contrast=1.0,
        min_size_mm=0.0,
        max_size_mm=300.0,
        min_length_mm=0,
        max_length_mm=0,
        min_width_mm=0,
        max_width_mm=0,
        blur_sz_mm=0.6,
        median_filter_sz_mm=0,
        downsample_factor=0.90,
        num_animals=num_animals,
        crop_dimension_mm=(width/PIX_PER_MM,height/PIX_PER_MM), 
        crop_offset_y_mm=0
    )
)
body_tracker = BodyTracker_CPU(
    BodyTrackerParamTracking(
        pix_per_mm=PIX_PER_MM,
        target_pix_per_mm=10,
        intensity=0.15,
        gamma=1.0,
        contrast=3.0,
        min_size_mm=2.0,
        max_size_mm=300.0,
        min_length_mm=0,
        max_length_mm=0,
        min_width_mm=0,
        max_width_mm=0,
        blur_sz_mm=0.6,
        median_filter_sz_mm=0,
        crop_dimension_mm=(5,5),
        crop_offset_y_mm=0
    )
)
eyes_tracker = EyesTracker_CPU(
    EyesTrackerParamTracking(
        pix_per_mm=PIX_PER_MM,
        target_pix_per_mm=40,
        thresh_lo=0.2,
        thresh_hi=0.8,
        gamma=2.0,
        dyntresh_res=5,
        contrast=5.0,
        size_lo_mm=0.1,
        size_hi_mm=30.0,
        blur_sz_mm=0.1,
        median_filter_sz_mm=0,
        crop_dimension_mm=(1,1.5),
        crop_offset_y_mm=-0.5
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
        tail_length_mm=3.0,
        blur_sz_mm=0.06,
        median_filter_sz_mm=0,
        contrast=3.0,
        gamma=0.75,
        crop_dimension_mm=(3.5,3.5),
        crop_offset_y_mm=3.5
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
        if DISPLAY:
            T_scale = SimilarityTransform2D.scaling(tracking['animals']['downsample_ratio']) 

            oly = overlay.overlay_global(tracking['animals']['image_downsampled'], tracking, T_scale)
            r = cv2.resize(oly,(512, 512))
            cv2.imshow('global',r)
            cv2.waitKey(1)
            
            fish = 0
            cv2.imshow('body_cropped', body_overlay.overlay_cropped(tracking['body'][fish]))
            cv2.waitKey(1)

            cv2.imshow('eyes_cropped', eyes_overlay.overlay_cropped(tracking['eyes'][fish]))
            cv2.waitKey(1)
            
            cv2.imshow('tail_cropped', tail_overlay.overlay_cropped(tracking['tail'][fish]))
            cv2.waitKey(1)

            cv2.imshow('body_resized', body_overlay.overlay_processed(tracking['body'][fish]))
            cv2.waitKey(1)

            cv2.imshow('eyes_resized', eyes_overlay.overlay_processed(tracking['eyes'][fish]))
            cv2.waitKey(1)
            
            cv2.imshow('tail_resized', tail_overlay.overlay_processed(tracking['tail'][fish]))
            cv2.waitKey(1)

finally:
    video_reader.close()
    cv2.destroyAllWindows()
