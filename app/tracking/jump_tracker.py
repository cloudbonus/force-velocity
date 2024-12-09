from dataclasses import dataclass
from enum import Enum

import mediapipe as mp
import numpy as np

from app.utils.video_source import VideoSource


class JumpState(Enum):
    TAKEOFF = 1
    LANDING = 2
    UNKNOWN = 3
    TRANSITION = 4


@dataclass
class JumpData:
    force: int
    velocity: int
    jump_state: JumpState
    timestamp: float


def read_landmark_positions_3d(results):
    if results.pose_landmarks is None or len(results.pose_landmarks) == 0:
        return None
    pose_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in results.pose_landmarks[0]])
    indices = np.array([23, 24, 31, 32])
    return pose_landmarks[indices]


class JumpForceVelocityTracker:
    def __init__(self, mass, video_path, model_path):
        self.mass = mass
        self.video_path = video_path
        self.model_path = model_path

        self.previous_position = None
        self.initial_ground = None
        self.previous_time = None
        self.previous_velocity = 0
        self.previous_force = 0
        self.previous_state = JumpState.UNKNOWN

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_segmentation_masks=True,

        )

        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self.video_source = VideoSource(self.video_path)

    def update(self):
        try:
            frame = next(self.video_source.stream_bgr())
        except StopIteration:
            return None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.data)
        results = self.pose_landmarker.detect_for_video(mp_image, int(frame.time * 1000))

        landmark_positions_3d = read_landmark_positions_3d(results)
        if landmark_positions_3d is None:
            return None

        current_time = frame.time
        force, velocity, state = self._compute(landmark_positions_3d, current_time)
        data_entry = JumpData(force=force, velocity=velocity, jump_state=state, timestamp=current_time)
        return data_entry

    def _compute(self, landmark_positions_3d, current_time):
        current_position = np.mean(landmark_positions_3d[:2, 1])
        ground = np.mean(landmark_positions_3d[2:, 1])

        if self.initial_ground is None:
            self.initial_ground = ground

        ground_change_percentage = abs(ground - self.initial_ground) / self.initial_ground
        if ground_change_percentage > 0.05:
            return self.previous_force, self.previous_velocity, JumpState.TRANSITION

        if self.previous_position is None:
            self.previous_position = current_position
            self.previous_time = current_time
            return 0, 0, JumpState.UNKNOWN

        delta_y = current_position - self.previous_position
        delta_t = current_time - self.previous_time

        if delta_t <= 0:
            return self.previous_force, self.previous_velocity, self.previous_state

        current_velocity = -delta_y / delta_t

        acceleration = (current_velocity - self.previous_velocity) / delta_t

        force = (
            self.mass * np.abs(acceleration) / np.abs(current_velocity)
            if np.abs(current_velocity) > 1e-5
            else self.mass * np.abs(acceleration)
        )

        error_margin = 0.001 * np.abs(self.previous_position)
        if current_position >= self.previous_position + error_margin:
            state = JumpState.LANDING
        elif current_position <= self.previous_position - error_margin:
            state = JumpState.TAKEOFF
        else:
            state = JumpState.UNKNOWN

        self.previous_position = current_position
        self.previous_velocity = current_velocity
        self.previous_time = current_time
        self.previous_force = force
        self.previous_state = state

        return force, current_velocity, state