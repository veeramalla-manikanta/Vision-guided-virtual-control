import cv2
import time
import pyautogui
from src.eye_tracker import EyeTracker
from src.hand_tracker import HandTracker
from src.virtual_keyboard import VirtualKeyboard

class Controller:
    def __init__(self, cam_index=0, cam_size=(640,480)):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_size[1])
        self.eye = EyeTracker(cam_size=cam_size)
        self.hand = HandTracker()
        self.vk = VirtualKeyboard(rows=3, cols=10)
        self.last_key = None
        self.key_cooldown = 0.6  # seconds between key presses
        self.last_key_time = 0

    def run(self):
        print("Press 'q' to quit. Ensure your terminal has permission to control the mouse/keyboard if required by OS.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            # 1) Eye tracking - move cursor
            cursor = self.eye.get_cursor_position(frame)
            if cursor:
                # move mouse (non-blocking). On some systems, this requires accessibility permissions.
                try:
                    pyautogui.moveTo(cursor[0], cursor[1], duration=0.05)
                except Exception:
                    pass

            # 2) Hand tracking - detect fingertip and show keyboard
            hands = self.hand.process(frame)
            if hands:
                # draw virtual keyboard overlay
                boxes = self.vk.draw(frame)
                # use first hand, index fingertip is landmark 8
                lm = hands[0]['landmarks']
                fingertip = lm[8]
                # convert normalized to px
                h, w, _ = frame.shape
                fx, fy = int(fingertip[0]*w), int(fingertip[1]*h)
                cv2.circle(frame, (fx, fy), 8, (0,255,0), -1)
                # check which key
                key = self.vk.fingertip_to_key(fingertip, frame.shape)
                # draw highlight
                if key:
                    cv2.putText(frame, f'Key: {key}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    now = time.time()
                    if key != self.last_key or (now - self.last_key_time) > self.key_cooldown:
                        # simulate key press
                        try:
                            pyautogui.press(key.lower())
                        except Exception:
                            pass
                        self.last_key = key
                        self.last_key_time = now
            else:
                # no hands: hide keyboard (we just draw nothing)
                pass

            cv2.imshow('Vision-Guided Virtual Control', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()