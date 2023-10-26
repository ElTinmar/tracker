import os
import socket
import pandas as pd
import numpy as np
import video_tools 
import tracker.trackers as trck
from image_tools import im2gray, im2single
from tqdm import tqdm

# get base folder location on different computers
DATA_LOCATION = {
    'hplaptop': '/home/martin/Downloads/Escapes/',
    'TheUgly': '/media/martin/MARTIN_8TB_0/Work/Baier/owncloud/Escapes/',
    'O1-596': '/home/martin/ownCloud - martin.privat@bi.mpg.de@owncloud.gwdg.de/Escapes/',
    'O1-619': '/home/martin/Documents/Escapes/',
    'L-O1-620': '/home/martinprivat/ownCloud/Escapes/',
}
host = socket.gethostname()
BASEFOLDER = DATA_LOCATION[host]

# relative path
FISHDATA = os.path.join(BASEFOLDER, 'fish.csv')
SELECT = [25]
DISPLAY = True

fish_data = pd.read_csv(
    FISHDATA, 
    usecols=['fish','video_file','timestamp_file','fov_size_mm']
)

for _, experiment in fish_data.iloc[SELECT,:].iterrows():
    fish = experiment['fish'] 
    video_file = os.path.join(BASEFOLDER, experiment['video_file']) 
    timestamp_file = os.path.join(BASEFOLDER, experiment['timestamp_file']) 
    fov_size_mm = experiment['fov_size_mm'] 
    print(f'Processing {fish}...')

    # video reader    
    crop = (0,0,600,600)
    reader = video_tools.OpenCV_VideoReader()
    reader.open_file(video_file, safe=False, crop=crop)
    num_frames = reader.get_number_of_frame()
    height = reader.get_height()
    width = reader.get_width()

    # background subtraction
    
    background = video_tools.StaticBackground(
        video_reader=reader
    )
    background.initialize()

    reader = video_tools.Buffered_OpenCV_VideoReader()
    reader.open_file(video_file, safe=False, crop=crop)
    reader.start()

    '''
    background = DynamicBackgroundMP(
        height=height,
        width=width,
        num_images = 500,
        every_n_image = 200
    )    
    '''

    #assignment = LinearSumAssignment(distance_threshold=50)
    LUT = np.zeros((600,600))
    assignment = trck.GridAssignment(LUT)
    accumulator = None

    display = video_tools.VideoDisplay(fps=10)
    display.start()

    # tracking 
    animal_tracker = trck.AnimalTracker(
        trck.AnimalTrackerParamTracking(
            pix_per_mm=40,
            target_pix_per_mm=7.5,
            animal_intensity=0.07,
            animal_norm=1.0,
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
        ),
        trck.AnimalTrackerParamOverlay()
    )
    body_tracker = trck.BodyTracker(
        trck.BodyTrackerParamTracking(
            pix_per_mm=40,
            target_pix_per_mm=7.5,
            body_intensity=0.25,
            body_norm=0.3,
            body_gamma=3.0,
            body_contrast=1.5,
            min_body_size_mm=2.0,
            max_body_size_mm=30.0,
            min_body_length_mm=2.0,
            max_body_length_mm=6.0,
            min_body_width_mm=0.4,
            max_body_width_mm=1.2,
            blur_sz_mm=1/7.5,
            median_filter_sz_mm=1/7.5,
        ),
        trck.BodyTrackerParamOverlay()
    )
    eyes_tracker = trck.EyesTracker(
        trck.EyesTrackerParamTracking(
            pix_per_mm=40,
            target_pix_per_mm=40,
            eye_norm=0.3,
            eye_gamma=3.0,
            eye_dyntresh_res=20,
            eye_contrast=1.5,
            eye_size_lo_mm=0.8,
            eye_size_hi_mm=10.0,
            blur_sz_mm=0.06,
            median_filter_sz_mm=0.06,
            crop_dimension_mm=(1.0,1.5),
            crop_offset_mm=-0.30
        ),
        trck.EyesTrackerParamOverlay()
    )
    tail_tracker = trck.TailTracker(
        trck.TailTrackerParamTracking(
            pix_per_mm=40,
            target_pix_per_mm=20,
            arc_angle_deg=120,
            n_tail_points=10,
            n_pts_arc=20,
            n_pts_interp=40,
            tail_length_mm=2.6,
            dist_swim_bladder_mm=0.2,
            blur_sz_mm=0.06,
            median_filter_sz_mm=0.06,
            tail_norm=0.2,
            tail_contrast=1.0,
            tail_gamma=0.75,
            crop_dimension_mm=(3.5,3.5),
            crop_offset_tail_mm=2.25
        ),
        trck.TailTrackerParamOverlay()
    )

    tracker = trck.Tracker(            
        assignment,
        accumulator,
        animal_tracker,
        body_tracker, 
        eyes_tracker, 
        tail_tracker
    )

    try:
        for i in tqdm(range(num_frames)):
            ret, image = reader.next_frame()
            if not ret:
                break
            img = im2single(im2gray(image))
            image_sub = background.subtract_background(img)
            tracking = tracker.track(image_sub)
            if DISPLAY:
                overlay = tracker.overlay(image, tracking)
                if overlay is not None:
                    display.queue_image(overlay)
    finally:
        reader.exit()
        reader.join()
        display.exit()
        display.join()

    