from video_tools import InMemory_OpenCV_VideoReader
from image_tools import im2single, im2gray
from tracker import (
    GridAssignment,
    MultiFishTracker_CPU, MultiFishOverlay_opencv, MultiFishTrackerParamTracking, MultiFishTrackerParamOverlay,
    AnimalTrackerKalman, AnimalOverlay_opencv, AnimalTrackerParamTracking, AnimalTrackerParamOverlay,
    BodyTrackerKalman, BodyOverlay_opencv, BodyTrackerParamTracking, BodyTrackerParamOverlay,
    EyesTrackerKalman, EyesOverlay_opencv, EyesTrackerParamTracking, EyesTrackerParamOverlay,
    TailTrackerKalman, TailOverlay_opencv, TailTrackerParamTracking, TailTrackerParamOverlay
)
from tqdm import tqdm
import numpy as np
import cv2
from geometry import SimilarityTransform2D
from tests.config import ANIMAL_PARAM, BODY_PARAM, EYES_PARAM, TAIL_PARAM

DISPLAY=True
DISPLAY_HEIGHT = 1024

# background subtracted video
INPUT_VIDEO = 'toy_data/multi_freelyswimming_1800x1800px_nobckg.avi'
PIX_PER_MM = 40

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

DISPLAY_WIDTH = int(width/height * DISPLAY_HEIGHT)

num_animals = 9
LUT = np.zeros((height, width))
LUT[0:600,0:600] = 0
LUT[0:600,600:1200] = 1
LUT[0:600,1200:1800] = 2
LUT[600:1200,0:600] = 3
LUT[600:1200,600:1200] = 4
LUT[600:1200,1200:1800] = 5
LUT[1200:1800,0:600] = 6
LUT[1200:1800,600:1200] = 7
LUT[1200:1800,1200:1800] = 8

assignment = GridAssignment(LUT, num_animals)

# tracking 
animal_tracker = AnimalTrackerKalman(
    assignment=assignment,
    tracking_param = AnimalTrackerParamTracking(
        pix_per_mm=PIX_PER_MM,
        num_animals=num_animals,
        **ANIMAL_PARAM
    ),
    fps = int(fps),
    model_order = 1
)
body_tracker = BodyTrackerKalman(
    tracking_param = BodyTrackerParamTracking(
        pix_per_mm=PIX_PER_MM,
        **BODY_PARAM
    ),
    fps = int(fps),
    model_order = 1
)
eyes_tracker = EyesTrackerKalman(
    tracking_param = EyesTrackerParamTracking(
        pix_per_mm=PIX_PER_MM,
        **EYES_PARAM
    ),
    fps = int(fps),
    model_order = 1
)
tail_tracker = TailTrackerKalman(
    tracking_param = TailTrackerParamTracking(
        pix_per_mm=PIX_PER_MM,
        **TAIL_PARAM
    ),
    fps = int(fps),
    model_order = 2
)

# overlay
animal_overlay = AnimalOverlay_opencv(AnimalTrackerParamOverlay())
body_overlay = BodyOverlay_opencv(BodyTrackerParamOverlay())
eyes_overlay = EyesOverlay_opencv(EyesTrackerParamOverlay())
tail_overlay = TailOverlay_opencv(TailTrackerParamOverlay())

tracker = MultiFishTracker_CPU(
    MultiFishTrackerParamTracking(
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
            r = cv2.resize(oly,(DISPLAY_HEIGHT, DISPLAY_WIDTH))
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
