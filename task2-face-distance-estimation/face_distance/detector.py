
import cv2


class FaceDetector:
    """Wraps OpenCV's built-in Haar Cascade face detector."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise IOError("Failed to load Haar Cascade face detector.")

    def detect(self, frame_bgr):
        """Returns the largest detected face as a dict, or None."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(60, 60),
        )

        if len(faces) == 0:
            return None

        # Pick the largest face (closest to camera) in case multiple are found
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cx = x + w / 2.0
        cy = y + h / 2.0

        return {"x": x, "y": y, "w": w, "h": h, "cx": cx, "cy": cy}
    