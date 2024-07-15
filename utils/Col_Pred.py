import math
import numpy as np
from deepsort_tracker import ObjectTracker
from main import ObjectDetection

class CollisionPrediction:
    def __init__(self, capture):
        self.object_detector = ObjectDetection(capture)
        self.object_tracker = ObjectTracker(capture)
        self.safety_margin = 2  # Adjust this value to set the desired safety margin (in meters)
        self.collision_threshold = 3  # Adjust this value to set the collision prediction threshold (in seconds)

    def calculate_time_to_collision(self, vehicle_a, vehicle_b):
        """
        Calculate the time-to-collision (TTC) between two vehicles.
        """
        x1, y1, w1, h1 = vehicle_a[0]
        x2, y2, w2, h2 = vehicle_b[0]
        
        # Calculate the center points of the vehicles
        center_a = np.array([x1 + w1 // 2, y1 + h1 // 2])
        center_b = np.array([x2 + w2 // 2, y2 + h2 // 2])
        
        # Calculate the relative speed and distance between the vehicles
        relative_speed = np.linalg.norm(vehicle_a[1] - vehicle_b[1])
        distance = np.linalg.norm(center_a - center_b)
        
        # Calculate the time-to-collision
        if relative_speed > 0:
            ttc = distance / relative_speed
        else:
            ttc = float('inf')
        
        return ttc

    def predict_collisions(self, detections, img):
        """
        Predict potential collisions between detected vehicles.
        """
        tracks = self.object_tracker.track_detect(detections, img)
        collision_warnings = []

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                track_a = tracks[i]
                track_b = tracks[j]

                if track_a.is_confirmed() and track_b.is_confirmed():
                    ttc = self.calculate_time_to_collision(track_a.to_ltrb(), track_b.to_ltrb())
                    if ttc < self.collision_threshold:
                        collision_warnings.append((track_a, track_b, ttc))

        return collision_warnings

    def alert_driver(self, collision_warnings, img):
        """
        Provide collision alerts to the driver.
        """
        for track_a, track_b, ttc in collision_warnings:
            x1, y1, x2, y2 = track_a.to_ltrb()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            
            # Add collision alert text to the image
            cvzone.putTextRect(img, f'Collision Warning! TTC: {ttc:.2f} s', (x1, y1 - 20), scale=1, thickness=1, colorR=(0, 0, 255))
            
            # Activate the vibrator or other alert mechanism here
            # ...

        return img

    def run(self):
        cap = cv2.VideoCapture(self.object_detector.capture)
        assert cap.isOpened()

        while True:
            _, img = cap.read()
            assert _
            detections, frames = self.object_detector.plot_boxes(self.object_detector.predict(img), img)
            collision_warnings = self.predict_collisions(detections, frames)
            alert_frame = self.alert_driver(collision_warnings, frames)
            cv2.imshow('Collision Avoidance', alert_frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collision_predictor = CollisionPrediction(capture=0)
    collision_predictor.run()