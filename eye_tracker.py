import cv2
import numpy as np
import pyautogui

class EyeTracker:
    def __init__(self, cam_size=(640, 480), screen_size=None, smoothing=0.2):
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.cam_w, self.cam_h = cam_size
        self.screen_w, self.screen_h = screen_size or pyautogui.size()
        self.smoothing = smoothing
        self.smoothed = None

    def detect_pupil(self, gray_eye):
        # Heuristic: threshold dark regions and find largest contour (pupil)
        blurred = cv2.GaussianBlur(gray_eye, (7,7), 0)
        _, th = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
        th = cv2.erode(th, None, iterations=2)
        th = cv2.dilate(th, None, iterations=3)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        if r < 2:
            return None
        return int(x), int(y)

    def get_cursor_position(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        # Prefer the largest detection (likely nearest eyes)
        if len(eyes) == 0:
            return None
        eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)
        ex, ey, ew, eh = eyes[0]
        eye_roi = gray[ey:ey+eh, ex:ex+ew]
        pupil = self.detect_pupil(eye_roi)
        if pupil is None:
            return None
        # map pupil pos inside eye ROI to screen coordinates
        px, py = pupil
        rel_x = px / float(ew)
        rel_y = py / float(eh)
        screen_x = int(rel_x * self.screen_w)
        screen_y = int(rel_y * self.screen_h)
        # smoothing
        if self.smoothed is None:
            self.smoothed = (screen_x, screen_y)
        else:
            sx = int(self.smoothed[0] * (1 - self.smoothing) + screen_x * self.smoothing)
            sy = int(self.smoothed[1] * (1 - self.smoothing) + screen_y * self.smoothing)
            self.smoothed = (sx, sy)
        return self.smoothed