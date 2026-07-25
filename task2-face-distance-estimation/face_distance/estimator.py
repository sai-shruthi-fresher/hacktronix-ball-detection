
import math


class DistanceEstimator:
    """Implements the pinhole-camera depth and angle formulas."""

    def __init__(self, focal_length_px: float, real_face_width_m: float = 0.15):
        self.f = focal_length_px
        self.W = real_face_width_m

    def estimate(self, face_center_x_px: float, face_width_px: float, image_center_x_px: float):
        if face_width_px <= 0:
            raise ValueError("face_width_px must be positive")

        # Depth: Z = (f * W) / w_px
        depth_m = (self.f * self.W) / face_width_px

        # Angle: theta = arctan((x - c_x) / f)
        angle_rad = math.atan((face_center_x_px - image_center_x_px) / self.f)
        angle_deg = math.degrees(angle_rad)

        return depth_m, angle_deg