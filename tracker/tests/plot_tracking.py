from video_tools import Buffered_OpenCV_VideoReader, VideoDisplay, FFMPEG_VideoWriter, OpenCV_VideoReader
from image_tools import im2single, im2gray
from tracker import (
    GridAssignment, MultiFishTracker_CPU, MultiFishOverlay_opencv, MultiFishTracking,
    AnimalTracker_CPU, AnimalOverlay_opencv, AnimalTrackerParamTracking, AnimalTrackerParamOverlay,
    BodyTracker_CPU, BodyOverlay_opencv, BodyTrackerParamTracking, BodyTrackerParamOverlay,
    EyesTracker_CPU, EyesOverlay_opencv, EyesTrackerParamTracking, EyesTrackerParamOverlay,
    TailTracker_CPU, TailOverlay_opencv, TailTrackerParamTracking, TailTrackerParamOverlay
)
from tqdm import tqdm
import numpy as np
from geometry import Affine2DTransform
from multiprocessing import Queue
from tracker import Accumulator
import pandas as pd

EXPORT_FPS = 100

class csv_saver(Accumulator):

    def __init__(self):
        self.x = []
        self.y = []
        self.fish_angle = []
        self.left_eye_angle = []
        self.right_eye_angle = []
        self.tail_tip_angle = []

    def update(self, tracking: MultiFishTracking):
        
        x = np.NaN
        y = np.NaN
        fish_angle = np.NaN
        left_eye_angle = np.NaN
        right_eye_angle = np.NaN
        tail_tip_angle = np.NaN

        if tracking is not None:
            if tracking.animals is not None:
                x = tracking.animals.centroids[0,0]
                y = tracking.animals.centroids[0,1]

            if (
                tracking.body is not None 
                and tracking.body[0] is not None
                and tracking.body[0].angle_rad is not None
            ):
                fish_angle = np.rad2deg(tracking.body[0].angle_rad)

            if tracking.eyes is not None and tracking.eyes[0] is not None:
                left_eye_angle = np.rad2deg(tracking.eyes[0].left_eye.angle)
                right_eye_angle = np.rad2deg(tracking.eyes[0].right_eye.angle)

            if (
                tracking.tail is not None
                and tracking.tail[0] is not None
                and tracking.tail[0].skeleton_interp is not None
            ):
                tail_tip_angle = np.rad2deg(
                    np.arctan2(
                        tracking.tail[0].skeleton_interp[-1,1] - tracking.tail[0].skeleton_interp[-2,1],
                        tracking.tail[0].skeleton_interp[-1,0] - tracking.tail[0].skeleton_interp[-2,0]
                    )
                )

        self.x.append(x)
        self.y.append(y)
        self.fish_angle.append(fish_angle)
        self.left_eye_angle.append(left_eye_angle)
        self.right_eye_angle.append(right_eye_angle)
        self.tail_tip_angle.append(tail_tip_angle)     

    def to_pandas(self):
        return pd.DataFrame.from_dict(
            {
                'x': self.x,
                'y': self.y,
                'fish_angle': self.fish_angle,
                'left_eye_angle': self.left_eye_angle,
                'right_eye_angle': self.right_eye_angle,
                'tail_tip_angle': self.tail_tip_angle
            }
        )

    def to_csv(self, filename: str):
        self.to_pandas().to_csv(filename)
        

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
accumulator = csv_saver()

q = Queue()
q_eyes = Queue()
q_tail = Queue()
display = VideoDisplay(q, fps=10, winname='tracking')
display_eyes = VideoDisplay(q_eyes, fps=10, winname='eyes')
display_tail = VideoDisplay(q_tail, fps=10, winname='tail')
display.start()
display_eyes.start()
display_tail.start()

# tracking 
animal_tracker = AnimalTracker_CPU(
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
body_tracker = BodyTracker_CPU(
    BodyTrackerParamTracking(
        pix_per_mm=40,
        target_pix_per_mm=20,
        body_intensity=0.20,
        body_brightness=0.0,
        body_gamma=1.0,
        body_contrast=3.0,
        min_body_size_mm=0.0,
        max_body_size_mm=30.0,
        min_body_length_mm=0.0,
        max_body_length_mm=12.0,
        min_body_width_mm=0.2,
        max_body_width_mm=6.0,
        blur_sz_mm=1/7.5,
        median_filter_sz_mm=1/7.5,
    )
)
eyes_tracker = EyesTracker_CPU(
    EyesTrackerParamTracking(
        pix_per_mm=40,
        target_pix_per_mm=120,
        eye_brightness=0.0,
        eye_gamma=3.0,
        eye_dyntresh_res=20,
        eye_contrast=5.0,
        eye_size_lo_mm=0.8,
        eye_size_hi_mm=10.0,
        blur_sz_mm=0.06,
        median_filter_sz_mm=0.06,
        crop_dimension_mm=(1.0,1.5),
        crop_offset_mm=-0.75
    )
)
tail_tracker = TailTracker_CPU(
    TailTrackerParamTracking(
        pix_per_mm=40,
        target_pix_per_mm=80,
        arc_angle_deg=120,
        n_tail_points=10,
        n_pts_arc=20,
        n_pts_interp=40,
        tail_length_mm=2.2,
        dist_swim_bladder_mm=0.0,
        blur_sz_mm=0.06,
        median_filter_sz_mm=0.06,
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
    max_num_animals=1,            
    assignment=assignment,
    accumulator=accumulator,
    animal=animal_tracker,
    body=body_tracker, 
    eyes=eyes_tracker, 
    tail=tail_tracker
)

overlay = MultiFishOverlay_opencv(
    animal_overlay,
    body_overlay,
    eyes_overlay,
    tail_overlay
)

video_writer = FFMPEG_VideoWriter(
    height=height,
    width=width,
    fps=EXPORT_FPS,
    filename='19-40-44_tracking.avi',
    codec = 'libx264',
    preset = 'medium'
)

video_writer_eyes= FFMPEG_VideoWriter(
    height=eyes_tracker.tracking_param.crop_dimension_px[1],
    width=eyes_tracker.tracking_param.crop_dimension_px[0],
    fps=EXPORT_FPS,
    filename='19-40-44_eyes_tracking.avi',
    codec = 'libx264',
    preset = 'medium'
)

video_writer_tail = FFMPEG_VideoWriter(
    height=tail_tracker.tracking_param.crop_dimension_px[1],
    width=tail_tracker.tracking_param.crop_dimension_px[0],
    fps=EXPORT_FPS,
    filename='19-40-44_tail_tracking.avi',
    codec = 'libx264',
    preset = 'medium'
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
        display.queue_image(oly)

        # save video
        video_writer.write_frame(oly[:,:,[2,1,0]])
        
        # display eyes 
        if tracking.eyes is not None and tracking.eyes[0] is not None:
            s = eyes_tracker.tracking_param.resize
            tx, ty = -tracking.eyes[0].offset
            S = Affine2DTransform.scaling(s,s)
            T = Affine2DTransform.translation(tx, ty)
            eye_oly = eyes_overlay.overlay(tracking.eyes[0].image, tracking.eyes[0], T @ S)
            display_eyes.queue_image(eye_oly)
            video_writer_eyes.write_frame(eye_oly[:,:,[2,1,0]])
        else:
            video_writer_eyes.write_frame(np.zeros((eyes_tracker.tracking_param.crop_dimension_px[1],eyes_tracker.tracking_param.crop_dimension_px[0],3), dtype=np.uint8))

        # display tail
        if tracking.tail is not None and tracking.tail[0] is not None:
            s = tail_tracker.tracking_param.resize
            tx, ty = -tracking.tail[0].offset
            S = Affine2DTransform.scaling(s,s)
            T = Affine2DTransform.translation(tx, ty)
            tail_oly = tail_overlay.overlay(tracking.tail[0].image, tracking.tail[0], T @ S)
            display_tail.queue_image(tail_oly)
            video_writer_tail.write_frame(tail_oly[:,:,[2,1,0]])
        else:
            video_writer_tail.write_frame(np.zeros((tail_tracker.tracking_param.crop_dimension_px[1],tail_tracker.tracking_param.crop_dimension_px[0],3), dtype=np.uint8))

finally:
    video_reader.exit()
    video_reader.join()
    display.exit()
    display_eyes.exit()
    display_tail.exit()
    display.join()
    display_eyes.join()
    display_tail.join()

accumulator.to_csv('tracking_results.csv')
video_writer.close()
video_writer_eyes.close()
video_writer_tail.close()


# plot

import matplotlib.pyplot as plt
import cv2

df = accumulator.to_pandas()
#df = pd.read_csv('tracking_results.csv')

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

t = np.arange(26000,27500)/fps
fig, ax = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(10, 5)
ax.set_facecolor((0,0,0))
plt.plot(t,smooth(df.left_eye_angle[26000:27500], 20), color= np.array(eyes_overlay.overlay_param.color_eye_left_BGR)[[2,1,0]]/255)
plt.plot(t,smooth(df.right_eye_angle[26000:27500], 20), color= np.array(eyes_overlay.overlay_param.color_eye_right_BGR)[[2,1,0]]/255)
plt.plot(t,np.array(df.tail_tip_angle[26000:27500]), color= np.array(tail_overlay.overlay_param.color_tail_BGR)[[2,1,0]]/255)
plt.savefig("tracking.svg")

# extract specific frame 
# NOTE safe = True is necessary here

frame = 27368

reader = OpenCV_VideoReader()
reader.open_file('19-40-44_tracking.avi', safe=True)
reader.seek_to(frame)
ret, img = reader.next_frame()
cv2.imshow('tracking', img)
cv2.waitKey(0)
cv2.imwrite(f'tracking_{frame}.png')

reader.open_file('19-40-44_eyes_tracking.avi', safe=True)
reader.seek_to(frame)
ret, img = reader.next_frame()
cv2.imshow('eyes', img)
cv2.waitKey(0)
cv2.imwrite(f'tracking_eyes_{frame}.png')


reader.open_file('19-40-44_tail_tracking.avi', safe=True)
reader.seek_to(frame)
ret, img = reader.next_frame()
cv2.imshow('tail',img)
cv2.waitKey(0)
cv2.imwrite(f'tracking_tail_{frame}.png')


cv2.destroyAllWindows()