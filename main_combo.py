import cv2
import mediapipe as mp
import pyautogui
pyautogui.FAILSAFE = False
import numpy as np
import joblib
import keyboard
import time
from threading import Thread
try:
    from playsound import playsound
except ImportError:
    def playsound(*args, **kwargs):
        pass  # fallback if playsound not installed

# Load trained eye model
model = joblib.load('eye_mouse_model.pkl')
reg_x_eye = model['reg_x_rf']
reg_y_eye = model['reg_y_rf']
RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398, 466]
SMOOTHING_WINDOW = 15
positions_buffer = []

# Hand tracking setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Screen size
screen_w, screen_h = pyautogui.size()

# Mode: 'eye' or 'hand'
mode = 'hand'

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

# For hand gesture state
pinch_active = False
right_click_active = False
last_pinch_time = 0
DOUBLE_CLICK_THRESHOLD = 0.4  # seconds
last_index_y = None  # For scroll detection
prev_mode = mode

# Eye blink detection state
blink_counter = 0
blink_clicked = False
last_blink_time = 0
BLINK_EAR_THRESHOLD = 0.20
BLINK_CONSEC_FRAMES = 2
BLINK_COOLDOWN = 0.5  # seconds
# For double blink (right click)
last_blink_event_time = 0
DOUBLE_BLINK_WINDOW = 0.7  # seconds

# Helper to play sound in background
def play_sound(path):
    try:
        Thread(target=playsound, args=(path,), daemon=True).start()
    except Exception as e:
        print(f'Sound error: {e}')

def run_eye_calibration(face_mesh, screen_w, screen_h):
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
    import csv
    with open('calib_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        cap2 = cv2.VideoCapture(0)
        for i, (sx, sy) in enumerate(calib_points):
            while True:
                ret, frame = cap2.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                msg = f"Calibration: Look at the {calib_labels[i]} corner and press SPACE"
                cv2.putText(frame, msg, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                cv2.circle(frame, (sx, sy), 20, (255,0,0), 3)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        h, w, _ = frame.shape
                        right_eye_indices = [362, 263, 387, 386, 385, 384, 398, 466]
                        eye_landmarks = []
                        for idx in right_eye_indices:
                            x = face_landmarks.landmark[idx].x * w
                            y = face_landmarks.landmark[idx].y * h
                            eye_landmarks.extend([x, y])
                        writer.writerow(list(eye_landmarks) + [sx, sy])
                        cv2.circle(frame, (int(eye_landmarks[0]), int(eye_landmarks[1])), 5, (0,0,255), -1)
                cv2.imshow('Calibration', frame)
                key = cv2.waitKey(1)
                if key == 32:  # SPACE
                    break
                elif key == ord('q'):
                    cap2.release()
                    cv2.destroyAllWindows()
                    return False
        cap2.release()
        cv2.destroyAllWindows()
    return True

while cap.isOpened():
    if keyboard.is_pressed('esc'):
        print('ESC pressed, quitting...')
        break
    if keyboard.is_pressed('e') and mode != 'eye':
        mode = 'eye'
        play_sound('assets/switch.wav')
    if keyboard.is_pressed('h') and mode != 'hand':
        mode = 'hand'
        play_sound('assets/switch.wav')
    # Reset hand state if mode changed
    if mode != prev_mode:
        pinch_active = False
        right_click_active = False
        last_index_y = None
        prev_mode = mode
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    overlay_msg = f"Mode: {'Eye' if mode == 'eye' else 'Hand'} (Press E/H to switch)"
    cv2.putText(frame, overlay_msg, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    # Get current mouse position for overlay
    mouse_x, mouse_y = pyautogui.position()
    cv2.circle(frame, (int(mouse_x * frame.shape[1] / screen_w), int(mouse_y * frame.shape[0] / screen_h)), 12, (0,255,0), 2)
    if mode == 'eye':
        if keyboard.is_pressed('c'):
            cv2.putText(frame, 'Calibration running...', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Eye/Hand Mouse Combo - Press E/H to switch, ESC to quit', frame)
            cv2.waitKey(500)
            print('Starting calibration...')
            if run_eye_calibration(face_mesh, screen_w, screen_h):
                import joblib
                model = joblib.load('eye_mouse_model.pkl')
                reg_x_eye = model['reg_x_rf']
                reg_y_eye = model['reg_y_rf']
                print('Calibration complete!')
            else:
                print('Calibration cancelled.')
            continue
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                eye_landmarks = []
                for idx in RIGHT_EYE_INDICES:
                    x = face_landmarks.landmark[idx].x * w
                    y = face_landmarks.landmark[idx].y * h
                    eye_landmarks.extend([x, y])
                X = np.array(eye_landmarks).reshape(1, -1)
                pred_x = int(reg_x_eye.predict(X)[0])
                pred_y = int(reg_y_eye.predict(X)[0])
                positions_buffer.append((pred_x, pred_y))
                if len(positions_buffer) > SMOOTHING_WINDOW:
                    positions_buffer.pop(0)
                avg_x = int(np.mean([p[0] for p in positions_buffer]))
                avg_y = int(np.mean([p[1] for p in positions_buffer]))
                avg_x = max(0, min(avg_x, screen_w - 1))
                avg_y = max(0, min(avg_y, screen_h - 1))
                pyautogui.moveTo(avg_x, avg_y, duration=0.2)
                cv2.circle(frame, (int(eye_landmarks[0]), int(eye_landmarks[1])), 5, (0,0,255), -1)
                # --- Blink detection (EAR) ---
                def euclidean(p1, p2):
                    return np.linalg.norm(np.array(p1) - np.array(p2))
                # Use right eye for blink
                top = [face_landmarks.landmark[386], face_landmarks.landmark[385]]
                bottom = [face_landmarks.landmark[374], face_landmarks.landmark[380]]
                left = face_landmarks.landmark[362]
                right = face_landmarks.landmark[263]
                vert = euclidean(
                    (top[0].x * w, top[0].y * h),
                    (bottom[0].x * w, bottom[0].y * h)
                )
                horz = euclidean(
                    (left.x * w, left.y * h),
                    (right.x * w, right.y * h)
                )
                ear = vert / horz if horz != 0 else 0
                # Blink logic
                if ear < BLINK_EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    now = time.time()
                    if blink_counter >= BLINK_CONSEC_FRAMES and not blink_clicked:
                        # Double blink detection
                        if now - last_blink_event_time < DOUBLE_BLINK_WINDOW:
                            pyautogui.click(button='right')
                            play_sound('assets/right_click.wav')
                            print('Eye mode: Double blink detected (right click)')
                            last_blink_event_time = 0  # reset
                        else:
                            pyautogui.click()
                            play_sound('assets/click.wav')
                            print('Eye mode: Blink detected (left click)')
                            last_blink_event_time = now
                        last_blink_time = now
                        blink_clicked = True
                    blink_counter = 0
                    blink_clicked = False
    elif mode == 'hand':
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            print('Hand detected')
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                # Index finger tip is landmark 8
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)
                # Map to screen
                screen_x = int(screen_w * hand_landmarks.landmark[8].x)
                screen_y = int(screen_h * hand_landmarks.landmark[8].y)
                pyautogui.moveTo(screen_x, screen_y, duration=0.2)
                cv2.circle(frame, (x, y), 10, (255,0,0), -1)
                # Pinch gesture (thumb tip 4 + index tip 8)
                thumb_tip = np.array([hand_landmarks.landmark[4].x * w, hand_landmarks.landmark[4].y * h])
                index_tip = np.array([hand_landmarks.landmark[8].x * w, hand_landmarks.landmark[8].y * h])
                middle_tip = np.array([hand_landmarks.landmark[12].x * w, hand_landmarks.landmark[12].y * h])
                pinch_dist = np.linalg.norm(thumb_tip - index_tip)
                two_finger_dist = np.linalg.norm(index_tip - middle_tip)
                # Thresholds (tune as needed)
                PINCH_THRESHOLD = 40
                TWO_FINGER_THRESHOLD = 40
                # Left click/drag and double click
                if pinch_dist < PINCH_THRESHOLD:
                    if not pinch_active:
                        now = time.time()
                        if now - last_pinch_time < DOUBLE_CLICK_THRESHOLD:
                            print('Double click detected')
                            play_sound('assets/click.wav')
                            pyautogui.doubleClick()
                        else:
                            print('Pinch (left click/drag) detected')
                            play_sound('assets/click.wav')
                            pyautogui.mouseDown()
                        last_pinch_time = now
                        pinch_active = True
                        last_index_y = y  # Start tracking for scroll
                    else:
                        # Scroll detection: if index finger moves up/down while pinching
                        if last_index_y is not None:
                            dy = y - last_index_y
                            SCROLL_THRESHOLD = 10  # pixels
                            if abs(dy) > SCROLL_THRESHOLD:
                                if dy < 0:
                                    print('Scroll up')
                                    pyautogui.scroll(30)  # Scroll up
                                else:
                                    print('Scroll down')
                                    pyautogui.scroll(-30)  # Scroll down
                                last_index_y = y
                else:
                    if pinch_active:
                        print('Pinch released')
                        pyautogui.mouseUp()
                        pinch_active = False
                        last_index_y = None
                # Right click (two-finger tap)
                if two_finger_dist < TWO_FINGER_THRESHOLD:
                    if not right_click_active:
                        print('Right click detected')
                        play_sound('assets/right_click.wav')
                        pyautogui.click(button='right')
                        right_click_active = True
                else:
                    right_click_active = False
        else:
            print('No hand detected')
    cv2.imshow('Eye/Hand Mouse Combo - Press E/H to switch, ESC to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 