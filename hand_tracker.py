import mediapipe as mp
import cv2

class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.6, tracking_conf=0.6):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=max_hands,
                                         min_detection_confidence=detection_conf,
                                         min_tracking_confidence=tracking_conf)

    def process(self, frame):
        # returns list of hands, each hand is dict with 'landmarks' (21 tuples normalized)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        hands_out = []
        h, w, _ = frame.shape
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = []
                for p in hand_landmarks.landmark:
                    lm.append((p.x, p.y))
                hands_out.append({'landmarks': lm})
        return hands_out