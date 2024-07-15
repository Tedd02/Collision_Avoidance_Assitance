from ultralytics import YOLO
import cv2
import math
import os
from utils.deepsort_tracker import ObjectTracker
from utils.distance_estimator import DistanceEstimation

class ObjectDetection:
    def __init__(self, capture):
        self.capture = capture
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):
        model = YOLO('./runs/detect/train5/weights/best.pt')
        model.fuse()
        return model

    def predict(self, img):
        results = self.model(img, stream=True)
        return results

    def plot_boxes(self, results, img):
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cls = int(box.cls[0])
                current_class = self.CLASS_NAMES_DICT[cls]
                conf = math.ceil(box.conf[0] * 100) / 100
                if conf > 0.5:
                    detections.append((([x1, y1, w, h]), conf, current_class))
        return detections, img

    def __call__(self):
        #cap = cv2.VideoCapture(self.capture)
        cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
        assert cap.isOpened()
        tracker = ObjectTracker(cap)

        while True:
            _, img = cap.read()
            assert _
            results = self.predict(img)
            detections, frames = self.plot_boxes(results, img)
            detect_frame = tracker.track_detect(detections, frames)
            cv2.imshow('Image', detect_frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

VIDEOS_DIR = os.path.join('.', 'out')
#video_path = os.path.join(VIDEOS_DIR, 'vid_2.MOV')
video_path = 0

detector = ObjectDetection(capture=video_path)
detector()