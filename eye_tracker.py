import numpy as np

# Mediapipe right eye landmark indices
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 466]

def get_right_eye_center(face_landmarks, frame_shape):
    h, w, _ = frame_shape
    xs = []
    ys = []
    for idx in RIGHT_EYE:
        x = int(face_landmarks.landmark[idx].x * w)
        y = int(face_landmarks.landmark[idx].y * h)
        xs.append(x)
        ys.append(y)
    return int(np.mean(xs)), int(np.mean(ys)) 