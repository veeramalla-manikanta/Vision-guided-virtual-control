import cv2

class VirtualKeyboard:
    def __init__(self, rows=3, cols=10, keys=None):
        # default QWERTY-like top rows (truncated to fit simple layout)
        default_keys = list('QWERTYUIOPASDFGHJKLZXCVBNM')
        self.keys = keys or default_keys
        self.rows = rows
        self.cols = cols
        self.layout = self._build_layout()

    def _build_layout(self):
        layout = []
        kidx = 0
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if kidx < len(self.keys):
                    row.append(self.keys[kidx])
                else:
                    row.append('')
                kidx += 1
            layout.append(row)
        return layout

    def draw(self, frame):
        h, w, _ = frame.shape
        key_w = w // self.cols
        key_h = int(h * 0.15) // self.rows if self.rows>0 else 50
        y0 = h - key_h * self.rows - 10
        boxes = []
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * key_w
                y = y0 + r * key_h
                label = self.layout[r][c]
                cv2.rectangle(frame, (x+2, y+2), (x+key_w-2, y+key_h-2), (200,200,200), 1)
                if label:
                    cv2.putText(frame, label, (x + 10, y + key_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                boxes.append({'r': r, 'c': c, 'x': x, 'y': y, 'w': key_w, 'h': key_h, 'label': label})
        return boxes

    def fingertip_to_key(self, fingertip, frame_shape):
        # fingertip is (x_norm, y_norm) normalized coordinates from mediapipe
        if fingertip is None:
            return None
        x_n, y_n = fingertip
        h, w = frame_shape[:2]
        px, py = int(x_n * w), int(y_n * h)
        boxes = self.draw_placeholder(frame_shape)
        for b in boxes:
            if b['x'] <= px <= b['x'] + b['w'] and b['y'] <= py <= b['y'] + b['h']:
                return b['label']
        return None

    def draw_placeholder(self, frame_shape):
        # helper to compute boxes without drawing
        h, w = frame_shape[:2]
        key_w = w // self.cols
        key_h = int(h * 0.15) // self.rows if self.rows>0 else 50
        y0 = h - key_h * self.rows - 10
        boxes = []
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * key_w
                y = y0 + r * key_h
                label = self.layout[r][c]
                boxes.append({'r': r, 'c': c, 'x': x, 'y': y, 'w': key_w, 'h': key_h, 'label': label})
        return boxes