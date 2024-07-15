
def calculate_speed(self, bbox, vehicle_speed):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        vspeed = 0

        if self.previous_detections and self.previous_time:
            current_time = time.time()
            time_elapsed = current_time - self.previous_time
            if time_elapsed > 0.001:
                for prev_bbox, prev_width, prev_height in self.previous_detections:
                    prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
                    prev_width = prev_x2 - prev_x1
                    prev_height = prev_y2 - prev_y1
                    distance_traveled = ((prev_x1 - x1) ** 2 + (prev_y1 - y1) ** 2) ** 0.5
                    # Adjust the distance_traveled based on the vehicle's speed
                    distance_traveled -= vehicle_speed * time_elapsed
                    speed = distance_traveled / time_elapsed
                    vspeed = speed
                    print(f"Object speed: {speed:.2f} m/s")
                    # You can use this speed information to predict possible collisions
                    # in your main `Out.py` file
                    break
            else:
                vspeed = 0.0
        else:
            vspeed = 0.0

        self.previous_detections.append((bbox, width, height))
        self.previous_time = time.time()

        return  vspeed

#Working function

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