import numpy as np
import cv2
import csv

class Calibrator:
    def __init__(self):
        self.eye_points = []  # [(eye_x, eye_y), ...]
        self.screen_points = []  # [(screen_x, screen_y), ...]
        self.calibrated = False
        self.transform = None
        self.csv_file = open('calib_data.csv', 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)

    def add_calibration_point(self, eye_point, screen_point, eye_landmarks=None):
        self.eye_points.append(eye_point)
        self.screen_points.append(screen_point)
        # Save for ML: eye_landmarks (flattened) + screen_point
        if eye_landmarks is not None:
            row = list(eye_landmarks) + list(screen_point)
            self.csv_writer.writerow(row)
            self.csv_file.flush()
        if len(self.eye_points) == 4:
            self.compute_transform()

    def compute_transform(self):
        # Use affine transform for 4 points
        eye = np.array(self.eye_points, dtype=np.float32)
        screen = np.array(self.screen_points, dtype=np.float32)
        self.transform = cv2.getPerspectiveTransform(eye, screen)
        self.calibrated = True

    def map_eye_to_screen(self, eye_point):
        if not self.calibrated or self.transform is None:
            return None
        pt = np.array([[list(eye_point)]], dtype=np.float32)  # shape (1,1,2)
        mapped = cv2.perspectiveTransform(pt, self.transform)
        return int(mapped[0,0,0]), int(mapped[0,0,1]) 