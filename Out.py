from ultralytics import YOLO
import cv2
import math
import os
import numpy as np
import random
from utils.deepsort_tracker import ObjectTracker
from utils.distance_estimator import DistanceEstimation

class ObjectDetection:
    def __init__(self, capture):
        self.capture = capture
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.deepsort_tracker = ObjectTracker(capture)

    def load_model(self):
        model = YOLO('./runs/detect/train5/weights/best.pt')
        model.fuse()
        return model

    def predict(self, img):
        results = self.model(img, stream=True)
        return results


    def plot_boxes(self, results, img):
        detections = []
        #speed = 10 + random.uniform(-2, 2)  # Speed in m/s
        for r in results:
            boxes = r.boxes
            for box in boxes:
                #x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cls = int(box.cls[0])
                current_class = self.CLASS_NAMES_DICT[cls]
                conf = math.ceil(box.conf[0] * 100) / 100
                if conf > 0.5:
                    detections.append((([x1, y1, w, h]), conf, current_class))
                    cv2.rectangle(img, (x1, y1), (x2, y2), (36, 255, 12), 2)
                    cv2.putText(img, current_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
                    vspeed = self.deepsort_tracker.calculate_speed([x1, y1, x2, y2])
                    cv2.putText(img, f"Speed: {vspeed:.2f} m/s", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
        return detections, img

    def __call__(self):
        if isinstance(self.capture, str):
            # If the capture is a video file, use it
            cap = cv2.VideoCapture(self.capture)
            ret, _ = cap.read()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert ret
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('./out/{}_out.MOV', fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
             # Perform camera calibration to get the camera matrix
            camera_matrix = np.array([[1000, 0, width/2], [0, 1000, height/2], [0, 0, 1]], dtype=np.float32)
            object_width = 1.8  # Average width of a car in meters
            distance_estimator = DistanceEstimation(camera_matrix, object_width)
        else:
            # Otherwise, use the default webcam
            cap = cv2.VideoCapture(self.capture)
            assert cap.isOpened()
            out = None

        tracker = ObjectTracker(cap)

        while True:
            ret, img = cap.read()
            resize = cv2.resize(img,(1080,960),interpolation=cv2.INTER_LINEAR)
            assert ret
            results = self.predict(resize)
            detections, frames = self.plot_boxes(results, resize)
            detect_frame = tracker.track_detect(detections, frames)
            print(detect_frame)
            for bbox, conf, current_class in detections:
                x, y, w, h = bbox
                distance = distance_estimator.estimate_distance([x, y, w, h])
                print(f"Distance to object: {distance:.2f} meters")                 
                cv2.putText(detect_frame[0],f'{distance:.2f} meters',(x, y+10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2) 
            cv2.imshow('Image', detect_frame[0])
            if out is not None:
                out.write(detect_frame[0])
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

VIDEOS_DIR = os.path.join('.', 'out')
video_path = os.path.join(VIDEOS_DIR, 'vid_3.MOV')
#video_path = 0  # Use 0 for the default webcam

detector = ObjectDetection(capture=video_path)
detector()