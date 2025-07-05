import cv2
import mediapipe as mp
import pyautogui
pyautogui.FAILSAFE = False  # Disable fail-safe for testing
import numpy as np
import keyboard  # For global hotkey
import time  # For blink click cooldown
from eye_tracker import get_right_eye_center
from calibrate import Calibrator

# --- Blink detection constants ---
RIGHT_EYE_LANDMARKS = [362, 263, 387, 386, 385, 384, 398, 466]
RIGHT_EYE_TOP = [386, 385]
RIGHT_EYE_BOTTOM = [374, 380]
BLINK_EAR_THRESHOLD = 0.23
BLINK_CONSEC_FRAMES = 3
BLINK_COOLDOWN = 0.5  # seconds

# --- Smoothing buffer ---
SMOOTHING_WINDOW = 15
positions_buffer = []

# --- Calibration ---
calibrator = Calibrator()
screen_w, screen_h = pyautogui.size()

# --- Mediapipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Blink detection state ---
blink_counter = 0
blink_clicked = False
last_blink_time = 0

# --- Calibration points (screen corners + sides + center) ---
calib_points = [
    (100, 100),  # Top-left
    (screen_w // 2, 100),  # Top-center
    (screen_w - 100, 100),  # Top-right
    (screen_w - 100, screen_h // 2),  # Right-center
    (screen_w - 100, screen_h - 100),  # Bottom-right
    (screen_w // 2, screen_h - 100),  # Bottom-center
    (100, screen_h - 100),  # Bottom-left
    (100, screen_h // 2),  # Left-center
    (screen_w // 2, screen_h // 2),  # Center
]
calib_labels = [
    "Top-Left", "Top-Center", "Top-Right",
    "Right-Center", "Bottom-Right", "Bottom-Center",
    "Bottom-Left", "Left-Center", "Center"
]

# --- Open webcam ---
cap = cv2.VideoCapture(0)

# --- Calibration loop ---
for i, (sx, sy) in enumerate(calib_points):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        msg = f"Look at the {calib_labels[i]} corner and press SPACE"
        cv2.putText(frame, msg, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.circle(frame, (sx, sy), 20, (255,0,0), 3)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                eye_x, eye_y = get_right_eye_center(face_landmarks, frame.shape)
                cv2.circle(frame, (eye_x, eye_y), 5, (0,0,255), -1)
        cv2.imshow('Calibration', frame)
        key = cv2.waitKey(1)
        if key == 32:  # SPACE pressed
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Collect all right eye landmark (x, y) for ML
                    h, w, _ = frame.shape
                    right_eye_indices = [362, 263, 387, 386, 385, 384, 398, 466]
                    eye_landmarks = []
                    for idx in right_eye_indices:
                        x = face_landmarks.landmark[idx].x * w
                        y = face_landmarks.landmark[idx].y * h
                        eye_landmarks.extend([x, y])
                    eye_x, eye_y = get_right_eye_center(face_landmarks, frame.shape)
                    calibrator.add_calibration_point((eye_x, eye_y), (sx, sy), eye_landmarks=eye_landmarks)
                    break
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
cv2.destroyAllWindows()

# --- Main loop ---
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
    overlay_msg = ""
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get right eye center
            eye_x, eye_y = get_right_eye_center(face_landmarks, frame.shape)
            # Smoothing
            positions_buffer.append((eye_x, eye_y))
            if len(positions_buffer) > SMOOTHING_WINDOW:
                positions_buffer.pop(0)
            avg_eye_x = int(np.mean([p[0] for p in positions_buffer]))
            avg_eye_y = int(np.mean([p[1] for p in positions_buffer]))
            print('Eye:', avg_eye_x, avg_eye_y)  # Debug print
            # Map to screen
            mapped = calibrator.map_eye_to_screen((avg_eye_x, avg_eye_y))
            print('Mapped:', mapped)  # Debug print
            if mapped:
                screen_x, screen_y = mapped
                # Clamp values to screen size
                screen_x = max(0, min(screen_x, screen_w - 1))
                screen_y = max(0, min(screen_y, screen_h - 1))
                pyautogui.moveTo(screen_x, screen_y, duration=0.2)
                cv2.circle(frame, (eye_x, eye_y), 5, (0,0,255), -1)
            # --- Blink detection (EAR) ---
            def euclidean(p1, p2):
                return np.linalg.norm(np.array(p1) - np.array(p2))
            h, w, _ = frame.shape
            top = [face_landmarks.landmark[idx] for idx in RIGHT_EYE_TOP]
            bottom = [face_landmarks.landmark[idx] for idx in RIGHT_EYE_BOTTOM]
            left = face_landmarks.landmark[362]
            right = face_landmarks.landmark[263]
            # EAR calculation
            vert = euclidean(
                (top[0].x * w, top[0].y * h),
                (bottom[0].x * w, bottom[0].y * h)
            )
            horz = euclidean(
                (left.x * w, left.y * h),
                (right.x * w, right.y * h)
            )
            ear = vert / horz if horz != 0 else 0
            print('EAR:', ear)  # Debug print for blink detection
            if ear < BLINK_EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= BLINK_CONSEC_FRAMES and not blink_clicked:
                    now = time.time()
                    if now - last_blink_time > BLINK_COOLDOWN:
                        pyautogui.click()
                        print('Blink click!')  # Debug print
                        overlay_msg = "Click!"
                        last_blink_time = now
                        blink_clicked = True
                blink_counter = 0
                blink_clicked = False
    if overlay_msg:
        cv2.putText(frame, overlay_msg, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    cv2.imshow('Eye Mouse - Press Q to Quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 