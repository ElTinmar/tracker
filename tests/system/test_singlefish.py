from video_tools import InMemory_OpenCV_VideoReader
from image_tools import im2single, im2gray
from tracker import (
    SingleFishTracker_CPU, SingleFishOverlay_opencv, SingleFishTrackerParamTracking, SingleFishTrackerParamOverlay,
    AnimalTracker_CPU, AnimalOverlay_opencv, AnimalTrackerParamTracking, AnimalTrackerParamOverlay,
    BodyTracker_CPU, BodyOverlay_opencv, BodyTrackerParamTracking, BodyTrackerParamOverlay,
    EyesTracker_CPU, EyesOverlay_opencv, EyesTrackerParamTracking, EyesTrackerParamOverlay,
    TailTracker_CPU, TailOverlay_opencv, TailTrackerParamTracking, TailTrackerParamOverlay
)
from tqdm import tqdm
import cv2
from geometry import SimilarityTransform2D
from tests.config import ANIMAL_PARAM, BODY_PARAM, EYES_PARAM, TAIL_PARAM

DISPLAY=True
DISPLAY_HEIGHT = 1024

# background subtracted video
VIDEOS = [
    ('toy_data/single_freelyswimming_504x500px_nobckg.avi', 40),
    ('toy_data/single_headembedded_544x380px_param_nobckg.avi', 90)
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

DISPLAY_WIDTH = int(width/height * DISPLAY_HEIGHT)

# tracking 
animal_tracker = AnimalTracker_CPU(
    tracking_param = AnimalTrackerParamTracking(
        pix_per_mm=PIX_PER_MM,
        crop_dimension_mm=(height/PIX_PER_MM, width/PIX_PER_MM),
        **ANIMAL_PARAM
    )
)
body_tracker = BodyTracker_CPU(
    tracking_param = BodyTrackerParamTracking(
        pix_per_mm=PIX_PER_MM,
        **BODY_PARAM
    ),
    fps = fps
)
eyes_tracker = EyesTracker_CPU(
    tracking_param = EyesTrackerParamTracking(
        pix_per_mm=PIX_PER_MM,
        **EYES_PARAM
    )
)
tail_tracker = TailTracker_CPU(
    tracking_param = TailTrackerParamTracking(
        pix_per_mm=PIX_PER_MM,
        **TAIL_PARAM
    )
)

# overlay
animal_overlay = AnimalOverlay_opencv(AnimalTrackerParamOverlay())
body_overlay = BodyOverlay_opencv(BodyTrackerParamOverlay())
eyes_overlay = EyesOverlay_opencv(EyesTrackerParamOverlay())
tail_overlay = TailOverlay_opencv(TailTrackerParamOverlay())

tracker = SingleFishTracker_CPU(
    SingleFishTrackerParamTracking(
        animal=animal_tracker,
        body=body_tracker, 
        eyes=eyes_tracker, 
        tail=tail_tracker
    )
)

overlay = SingleFishOverlay_opencv(
    SingleFishTrackerParamOverlay(
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
            
            cv2.imshow('body_cropped', body_overlay.overlay_cropped(tracking['body']))
            cv2.waitKey(1)

            cv2.imshow('eyes_cropped', eyes_overlay.overlay_cropped(tracking['eyes']))
            cv2.waitKey(1)
            
            cv2.imshow('tail_cropped', tail_overlay.overlay_cropped(tracking['tail']))
            cv2.waitKey(1)

            cv2.imshow('body_resized', body_overlay.overlay_processed(tracking['body']))
            cv2.waitKey(1)

            cv2.imshow('eyes_resized', eyes_overlay.overlay_processed(tracking['eyes']))
            cv2.waitKey(1)
            
            cv2.imshow('tail_resized', tail_overlay.overlay_processed(tracking['tail']))
            cv2.waitKey(1)

finally:
    video_reader.close()
    cv2.destroyAllWindows()
