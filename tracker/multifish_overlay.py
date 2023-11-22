import numpy as np
import cv2
from numpy.typing import NDArray
from typing import Dict, Optional
import .animal_overlay
import .body_overlay
import .eyes_overlay
import .tail_overlay

def overlay_local(image: NDArray, tracking: Optional[Dict]) -> NDArray:

    if tracking is None:
        return None
    
    # copy to avoid side-effects
    image = tracking['image'].copy()
    image = np.dstack((image,image,image))

    # loop over animals
    for idx, id in zip(tracking['indices'], tracking['identities']):
        if tracking['animals'] is not None:

            # overlay animal bounding boxes
            image = self.animal_tracker.overlay(image, tracking['animals'])
            
            # translate according to animal position 
            bbox_bottomleft = tracking['animals'].bounding_boxes[idx,:2]

            if (self.body_tracker is not None)  and (tracking['body'][id] is not None):
                # rotate according to animal orientation 
                angle = tracking['body'][id].angle_rad
                rotation = rotation_matrix(np.rad2deg(angle))[:2,:2]
                
                # overlay body
                image = self.body_tracker.overlay(image, tracking['body'][id], bbox_bottomleft)
                
                # overlay eyes
                if (self.eyes_tracker is not None)  and (tracking['eyes'][id]is not None):
                    offset_eye_ROI = bbox_bottomleft + tracking['body'][id].centroid 
                    image = self.eyes_tracker.overlay(image, tracking['eyes'][id], offset_eye_ROI, rotation)
                
                # overlay tail
                if (self.tail_tracker is not None)  and (tracking['tail'][id] is not None):
                    offset_tail_ROI = bbox_bottomleft + tracking['body'][id].centroid 
                    image = self.tail_tracker.overlay(image, tracking['tail'][id], offset_tail_ROI, rotation)

        # show ID
        cv2.putText(image, str(id), bbox_bottomleft.astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv2.LINE_AA)
    
    return image



def overlay(self, image, tracking):
    if tracking is None:
        return None
    
    # loop over animals
    for idx, id in zip(tracking['indices'], tracking['identities']):
        if (self.animal_tracker is not None)  and (tracking['animals'] is not None):
            # overlay animal bounding boxes
            image = self.animal_tracker.overlay(image, tracking['animals'])
            
            # translate according to animal position 
            bbox_bottomleft = tracking['animals'].bounding_boxes[idx,:2]

            if (self.body_tracker is not None)  and (tracking['body'][id] is not None):
                # rotate according to animal orientation 
                angle = tracking['body'][id].angle_rad
                rotation = rotation_matrix(np.rad2deg(angle))[:2,:2]
                
                # overlay body
                image = self.body_tracker.overlay(image, tracking['body'][id], bbox_bottomleft)
                
                # overlay eyes
                if (self.eyes_tracker is not None)  and (tracking['eyes'][id] is not None):
                    offset_eye_ROI = bbox_bottomleft + tracking['body'][id].centroid 
                    image = self.eyes_tracker.overlay(image, tracking['eyes'][id], offset_eye_ROI, rotation)
                
                # overlay tail
                if (self.tail_tracker is not None)  and (tracking['tail'][id] is not None):
                    offset_tail_ROI = bbox_bottomleft + tracking['body'][id].centroid 
                    image = self.tail_tracker.overlay(image, tracking['tail'][id], offset_tail_ROI, rotation)

        # show ID
        cv2.putText(image, str(id), bbox_bottomleft.astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv2.LINE_AA)
    
    return image


