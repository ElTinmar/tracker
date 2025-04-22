from video_tools import InMemory_OpenCV_VideoReader
from image_tools import im2single, im2gray
from tracker import (
    SingleFishTracker_CPU, SingleFishTrackerParamTracking,
    AnimalTracker_CPU , AnimalTrackerParamTracking,
    BodyTracker_CPU,  BodyTrackerParamTracking,
    EyesTracker_CPU,  EyesTrackerParamTracking,
    TailTracker_CPU,  TailTrackerParamTracking
)
import timeit
from functools import partial
import numpy as np
from tests.config import ANIMAL_PARAM, BODY_PARAM, EYES_PARAM, TAIL_PARAM

NREP = 10
NFRAMES = 100

# background subtracted video
VIDEOS = [
    ('toy_data/single_freelyswimming_504x500px_nobckg.avi', 40),
    ('toy_data/single_headembedded_544x380px_noparam_nobckg.avi', 130),
    ('toy_data/single_headembedded_544x380px_param_nobckg.avi', 90)
]

def load_video(video_num: int = 0):

    video, pix_per_mm = VIDEOS[video_num]

    video_reader = InMemory_OpenCV_VideoReader()
    video_reader.open_file(
        filename = video, 
        memsize_bytes = 4e9, 
        safe = False, 
        single_precision = True, 
        grayscale = True
    )

    height = video_reader.get_height()
    width = video_reader.get_width()
    fps = video_reader.get_fps()  
    num_frames = video_reader.get_number_of_frame()

    # tracking 
    animal_tracker = AnimalTracker_CPU(
        tracking_param = AnimalTrackerParamTracking(
            pix_per_mm=pix_per_mm,
            **ANIMAL_PARAM
        )
    )
    body_tracker = BodyTracker_CPU(
        tracking_param = BodyTrackerParamTracking(
            pix_per_mm=pix_per_mm,
            **BODY_PARAM
        )
    )
    eyes_tracker = EyesTracker_CPU(
        tracking_param = EyesTrackerParamTracking(
            pix_per_mm=pix_per_mm,
            **EYES_PARAM
        )
    )
    tail_tracker = TailTracker_CPU(
        tracking_param = TailTrackerParamTracking(
            pix_per_mm=pix_per_mm,
            **TAIL_PARAM
        )
    )

    tracker = SingleFishTracker_CPU(
        SingleFishTrackerParamTracking(
            animal=animal_tracker,
            body=body_tracker, 
            eyes=eyes_tracker, 
            tail=tail_tracker
        )
    )

    return video_reader, tracker, height, width

def track(video_reader, tracker) -> None:

    for i in range(NFRAMES):

        (rval, frame) = video_reader.next_frame()
        if not rval:
            raise RuntimeError('VideoReader was unable to read the whole video')
        
        # convert
        frame_gray = im2single(im2gray(frame))

        # track
        tracking = tracker.track(frame_gray)

if __name__ == '__main__':

    for id, vid in enumerate(VIDEOS):
        video_reader, tracker, height, width = load_video(id)
        track_fun = partial(track, video_reader=video_reader, tracker=tracker)
        timings = NFRAMES/np.array(timeit.repeat(track_fun, number=1, repeat=NREP))
        avg = np.mean(timings)
        std = np.std(timings)
        print(f'FPS: {avg:.2f} \N{PLUS-MINUS SIGN} {std:.2f}, ({height},{width}), {vid}')
        video_reader.close()
