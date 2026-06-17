from video_tools import InMemory_OpenCV_VideoReader
from image_tools import im2single, im2gray
from tracker import (
    SingleFishTracker_CPU, SingleFishOverlay_opencv, SingleFishTrackerParamTracking, SingleFishTrackerParamOverlay,
    AnimalTracker_CPU, AnimalOverlay_opencv, AnimalTrackerParamTracking, AnimalTrackerParamOverlay,
    BodyTracker_CPU, BodyOverlay_opencv, BodyTrackerParamTracking, BodyTrackerParamOverlay,
    EyesTracker_CPU, EyesOverlay_opencv, EyesTrackerParamTracking, EyesTrackerParamOverlay,
    TailTracker_CPU, TailOverlay_opencv, TailTrackerParamTracking, TailTrackerParamOverlay,
    HeadEmbeddedTracker_CPU, HeadEmbeddedOverlay_opencv, HeadEmbedded_ParamTracking, HeadEmbedded_ParamOverlay,
    LighthillPredictor
)
from tqdm import tqdm
import cv2
from geometry import SimilarityTransform2D
from tests.config import ANIMAL_PARAM, BODY_PARAM, EYES_PARAM, TAIL_PARAM
import numpy as np
from qt_widgets import imshow, waitKey, destroyAllWindows

DISPLAY=True
DISPLAY_HEIGHT = 512

# background subtracted video
VIDEOS = [
    ('toy_data/single_freelyswimming_504x500px_nobckg.avi', 40, None),
    ('toy_data/single_headembedded_544x380px_param_nobckg.avi', 90, None),
    ('toy_data/freely_swimming_param.mp4', 40, 'toy_data/freely_swimming_param.png')
]

# background subtracted video
VIDEO_NUM = 0
INPUT_VIDEO, PIX_PER_MM, BCKG_FILE = VIDEOS[VIDEO_NUM]

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

if BCKG_FILE is not None:
    img = cv2.imread(BCKG_FILE)
    background_image = im2single(im2gray(img))
else:
    background_image = np.zeros((height, width), np.float32)

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

predictor = LighthillPredictor(forward_gain=0.062, angular_gain=0.0092, framerate=fps)
head_embedded_tracker = HeadEmbeddedTracker_CPU(
    tracking_param = HeadEmbedded_ParamTracking(
        tail=tail_tracker,
        position_predictor=predictor
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
head_embedded_overlay = HeadEmbeddedOverlay_opencv(
    HeadEmbedded_ParamOverlay(tail_overlay)
)


data = np.zeros((num_frames,3), np.float32)
pred = np.zeros((num_frames,3), np.float32)

try:
    for i in tqdm(range(num_frames)):
        (rval, frame) = video_reader.next_frame()
        if not rval:
            raise RuntimeError('VideoReader was unable to read the whole video')
        
        # convert
        #frame_gray = im2single(im2gray(frame))

        # track
        tracking = tracker.track(frame, background_image)
        head_embedded_tracking = head_embedded_tracker.track(tracking['tail']['image_cropped'])
        
        data[i,:2] = tracking['body']['centroid_global']
        data[i,2] = tracking['body']['angle_rad_global']

        pred[i,0] = head_embedded_tracking['predicted_x']
        pred[i,1] = head_embedded_tracking['predicted_y']
        pred[i,2] = head_embedded_tracking['predicted_theta']

        # display tracking
        if DISPLAY:
            T_scale = SimilarityTransform2D.scaling(tracking['animals']['downsample_ratio']) 
            oly = overlay.overlay_global(tracking['animals']['image_downsampled'], tracking, T_scale)
            r = cv2.resize(oly,(DISPLAY_HEIGHT, DISPLAY_WIDTH))
            # cv2.imshow('frame', frame)
            imshow('global',r)
            # cv2.imshow('body_cropped', body_overlay.overlay_cropped(tracking['body']))
            # cv2.imshow('eyes_cropped', eyes_overlay.overlay_cropped(tracking['eyes']))
            # cv2.imshow('tail_cropped', tail_overlay.overlay_cropped(tracking['tail']))
            # cv2.imshow('body_resized', body_overlay.overlay_processed(tracking['body']))
            # cv2.imshow('eyes_resized', eyes_overlay.overlay_processed(tracking['eyes']))

            oly = tail_overlay.overlay_processed(tracking['tail'])
            r = cv2.resize(oly,(DISPLAY_HEIGHT, DISPLAY_WIDTH))
            imshow('tail_resized', r)
            waitKey(1)

finally:
    video_reader.close()
    destroyAllWindows()

# transform coordinates
data = data - data[0]
data[:,2] = np.unwrap(data[:,2])

import matplotlib.pyplot as plt
plt.plot(data)
plt.show()

plt.plot(pred)
plt.show()

plt.figure()
plt.plot(data[:,0])
plt.plot(pred[:,0])
plt.show(block=False)

plt.figure()
plt.plot(data[:,1])
plt.plot(pred[:,1])
plt.show(block=False)

plt.figure()
plt.plot(data[:,2])
plt.plot(pred[:,2])
plt.show(block=False)