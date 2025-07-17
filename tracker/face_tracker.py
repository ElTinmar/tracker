import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from geometry import SimilarityTransform2D
from image_tools import im2gray
from .core import Tracker

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_alt2.xml")
eyes_classifier =  cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye_tree_eyeglasses.xml")
dt = np.dtype([
    ('success', np.bool_),
    ('centroid_global', np.float32, (2,)),
    ('body_axes_global', np.float32, (2,2)),
])
failure = np.array((
    False, 
    np.zeros((2,), np.float32),
    np.zeros((2,2), np.float32),
    ), 
    dtype=dt
)

class FaceTracker(Tracker):

    def track(
        self, 
        image: NDArray, 
        background_image: Optional[NDArray] = None,
        centroid: Optional[NDArray] = None,
        T_input_to_global: SimilarityTransform2D = SimilarityTransform2D.identity()
    ) -> NDArray:

        image = im2gray(image)     
                
        # detect face
        faces = face_classifier.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
        if len(faces) != 1:
            print(f'faces detected {len(faces)}')
            return failure
               
        x, y, w, h = faces[0]
        faceROI = image[y:y+h,x:x+w]
        eyes = eyes_classifier.detectMultiScale(faceROI)
        if len(eyes) != 2:
            print(f'eyes detected {len(eyes)}')
            return failure
        
        eye_centroids = np.zeros((2,2), dtype=np.float32)
        for i, (x2,y2,w2,h2) in enumerate(eyes):
            eye_centroids[i,:] = (x + x2 + w2//2, y + y2 + h2//2)
        
        # find which eye is which
        left_eye_index = np.argmin(eye_centroids[:,0])
        left_eye = eye_centroids[left_eye_index,:]
        right_eye = eye_centroids[1-left_eye_index,:]

        midpoint = np.mean(eye_centroids, axis=0)
        lateral_axis = right_eye-midpoint
        lateral_axis = lateral_axis / np.linalg.norm(lateral_axis)
        vertical_axis = np.array((-lateral_axis[1], lateral_axis[0]))
        heading = np.column_stack((vertical_axis, lateral_axis))
   
        return np.array((True, midpoint, heading), dtype=dt)
    

if __name__ == "__main__":

    cam = cv2.VideoCapture(0)
    tracker = FaceTracker()
    for i in range(500):
        ret, frame = cam.read()
        res = tracker.track(frame)

        c = (
            int(res['centroid_global'][0]), 
            int(res['centroid_global'][1])
        )
        x = (
            int(res['centroid_global'][0] + 100*res['body_axes_global'][0,0]),
            int(res['centroid_global'][1] + 100*res['body_axes_global'][1,0]),
        )
        y = (
            int(res['centroid_global'][0] + 100*res['body_axes_global'][0,1]),
            int(res['centroid_global'][1] + 100*res['body_axes_global'][1,1]),
        )
        frame = cv2.circle(frame, c, 5, (0,0,255), -1)
        frame = cv2.line(frame, c, x, (0,0,255))
        frame = cv2.line(frame, c, y, (0,0,255))
        cv2.imshow('face_tracking', frame)
        cv2.waitKey(1)

    cam.release()