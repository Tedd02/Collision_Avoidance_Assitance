import cv2
import cvzone
import math
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self, capture):
        self.capture = capture
        self.tracker = self.create_tracker()

    def create_tracker(self):
        tracker = DeepSort(max_age=5, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.3, nn_budget=None, override_track_class=None, embedder='mobilenet', half=True, bgr=True, embedder_gpu=True, embedder_model_name=None, embedder_wts=None, polygon=False, today=True)
        return tracker

    def track_detect(self, detections, img):
        tracks = self.tracker.update_tracks(detections, frame=img)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            bbox = ltrb
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.putTextRect(img, f'ID: {track_id}', (x1, y1), scale=1, thickness=1, colorR=(0, 0, 255))
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=1, colorR=(255, 0, 255))
        return img