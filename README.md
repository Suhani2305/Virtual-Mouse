# Virtual Mouse: Eye & Hand Controlled

**Control your computer with your eyes or your hand!  
Switch modes anytime. Designed for accessibility, fun, and futuristic interaction.**

---

## ✨ Features

- **Eye Mode:** Move mouse with your gaze (ML-powered, personalized calibration)
- **Hand Mode:** Move mouse with your index finger, click/drag with pinch, right click with two-finger tap, scroll with pinch+move
- **Mode Switch:** Press `E` for Eye, `H` for Hand, `ESC` to quit
- **Sound Feedback:** Click, right click, and mode switch sounds
- **Visual Pointer:** Green circle shows current mouse position
- **Multi-point Calibration:** For accurate, smooth control
- **Easy to use, open-source, and privacy-friendly**

---

## 🧑‍💻 Why Virtual Mouse?

- **Accessibility:** For people with paralysis, limited mobility, or anyone who wants hands-free or gesture-based control
- **Innovation:** Combines computer vision, ML, and gesture control in one project
- **Fun & Futuristic:** Try it, demo it, or build on top of it!

---

## 🚀 Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/Suhani2305/VirtualMouse.git
cd VirtualMouse
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Add Sound Files
- Download free sounds from [freesound.org](https://freesound.org/) or use the links in the README above.
- Place `click.wav`, `right_click.wav`, `switch.wav` in the `assets/` folder.

### 4. Calibrate (First Time)
```bash
python main.py
```
- Look at each calibration point and press SPACE.

### 5. Train the ML Model
```bash
python train_eye_mouse_model.py
```

### 6. Run the Combo Mouse
```bash
python main_combo.py
```
- Default: Hand mode (move/click/scroll with hand)
- Press `E` for Eye mode, `H` for Hand mode, `ESC` to quit

---

## 🖱️ Hand Mode Controls

- **Move:** Index finger tip
- **Left Click/Drag:** Pinch (thumb + index tip)
- **Double Click:** Double pinch (quickly)
- **Right Click:** Index + middle finger tap
- **Scroll:** Pinch hold + move index finger up/down

## 👁️ Eye Mode Controls

- **Move:** Gaze direction (after calibration)
- **Recalibrate:** Press `C` anytime in eye mode

---

## 📢 Credits

- Built with [OpenCV](https://opencv.org/), [MediaPipe](https://mediapipe.dev/), [scikit-learn](https://scikit-learn.org/), [pyautogui](https://pyautogui.readthedocs.io/), [playsound](https://github.com/TaylorSMarks/playsound)
- Free sounds from [freesound.org](https://freesound.org/)
- Made with ❤️ for accessibility and innovation by Suhani

---

 
