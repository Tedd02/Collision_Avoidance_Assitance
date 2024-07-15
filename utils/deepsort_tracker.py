import cv2
import cvzone
import math
import time
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self, capture):
        self.capture = capture
        self.tracker = self.create_tracker()
        self.previous_detections = []
        self.previous_time = None

    def create_tracker(self):
        tracker = DeepSort(max_age=5, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.3, nn_budget=None, override_track_class=None, embedder='mobilenet', half=True, bgr=True, embedder_gpu=True, embedder_model_name=None, embedder_wts=None, polygon=False, today=True)
        return tracker
    
    def calculate_speed(self, bbox):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        vspeed = 0

        if self.previous_detections and self.previous_time:
            current_time = time.time()
            time_elapsed = current_time - self.previous_time
            if time_elapsed > 0.001:  # Threshold to avoid division by zero
                for prev_bbox, prev_width, prev_height in self.previous_detections:
                    prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
                    prev_width = prev_x2 - prev_x1
                    prev_height = prev_y2 - prev_y1
                    distance_traveled = ((prev_x1 - x1) ** 2 + (prev_y1 - y1) ** 2) ** 0.5
                    speed = distance_traveled / time_elapsed
                    vspeed = speed
                    print(f"Object speed: {speed:.2f} m/s")
                    # You can use this speed information to predict possible collisions
                    # in your main `Out.py` file
                    break  # Assuming only one previous detection is used for speed calculation
            else:
                vspeed = 0.0  # Default speed value if time_elapsed is too small

        self.previous_detections.append((bbox, width, height))
        self.previous_time = time.time()

        return vspeed

    def track_detect(self, detections, img):
        tracks = self.tracker.update_tracks(detections, frame=img)
        id = 0
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            id = track_id
            ltrb = track.to_ltrb()
            bbox = ltrb
            self.calculate_speed(bbox)
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            #cvzone.putTextRect(img, f'ID: {track_id}', (x1, y1), scale=1, thickness=1, colorR=(0, 0, 255))
            #cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=1, colorR=(255, 0, 255))    
        return img, id