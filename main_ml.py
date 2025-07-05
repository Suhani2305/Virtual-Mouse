import cv2
import mediapipe as mp
import pyautogui
pyautogui.FAILSAFE = False
import numpy as np
import joblib
import keyboard

# Load trained model
model = joblib.load('eye_mouse_model.pkl')
# Use RandomForestRegressor for prediction (more accurate)
reg_x = model['reg_x_rf']
reg_y = model['reg_y_rf']
# If you want to use LinearRegression instead, use reg_x_lin/reg_y_lin
# reg_x = model['reg_x_lin']
# reg_y = model['reg_y_lin']

# Right eye landmark indices
RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398, 466]

# Smoothing
SMOOTHING_WINDOW = 15
positions_buffer = []

# Get screen size
screen_w, screen_h = pyautogui.size()

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    if keyboard.is_pressed('esc'):
        print('ESC pressed, quitting...')
        break
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            eye_landmarks = []
            for idx in RIGHT_EYE_INDICES:
                x = face_landmarks.landmark[idx].x * w
                y = face_landmarks.landmark[idx].y * h
                eye_landmarks.extend([x, y])
            # Predict screen position
            X = np.array(eye_landmarks).reshape(1, -1)
            pred_x = int(reg_x.predict(X)[0])
            pred_y = int(reg_y.predict(X)[0])
            # Smoothing
            positions_buffer.append((pred_x, pred_y))
            if len(positions_buffer) > SMOOTHING_WINDOW:
                positions_buffer.pop(0)
            avg_x = int(np.mean([p[0] for p in positions_buffer]))
            avg_y = int(np.mean([p[1] for p in positions_buffer]))
            # Clamp to screen
            avg_x = max(0, min(avg_x, screen_w - 1))
            avg_y = max(0, min(avg_y, screen_h - 1))
            pyautogui.moveTo(avg_x, avg_y, duration=0.2)
            # Visual feedback
            cv2.circle(frame, (int(eye_landmarks[0]), int(eye_landmarks[1])), 5, (0,0,255), -1)
    cv2.imshow('ML Eye Mouse - Press ESC to Quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 