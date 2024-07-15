import cv2
import numpy as np

import numpy as np

class DistanceEstimation:
    def __init__(self, camera_matrix, object_width):
        self.camera_matrix = camera_matrix
        self.object_width = object_width

    def estimate_distance(self, bbox):
        """
        Estime la distance entre le véhicule équipé de la caméra et l'objet dans la boîte englobante (bbox).
        
        Paramètres:
        bbox (list): La boîte englobante au format [x, y, width, height].
        
        Retourne:
        float: La distance estimée en mètres.
        """
        x, y, w, h = bbox
        
        # Calcul de la distance à l'aide de la formule de la triangulation
        focal_length = self.camera_matrix[0, 0]
        object_height_in_frame = h
        distance = (self.object_width * focal_length) / object_height_in_frame
        
        return distance