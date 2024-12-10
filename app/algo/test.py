import time

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from enum import Enum

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
    if not results or not results.pose_landmarks:
        return None
    indices = [23, 24, 31, 32]
    pose_landmarks = np.array(
        [(lm.x, lm.y, lm.z) for i, lm in enumerate(results.pose_landmarks.landmark) if i in indices]
    )
    return pose_landmarks

class JumpForceVelocityTracker:
    def __init__(self, mass):
        self.mass = mass
        self.previous_position = None
        self.initial_ground = None
        self.previous_time = None
        self.previous_velocity = 0
        self.previous_force = 0
        self.previous_state = JumpState.UNKNOWN

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def update(self, frame, timestamp):
        mp_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(mp_image)
        landmark_positions_3d = read_landmark_positions_3d(results)
        if landmark_positions_3d is None:
            return None

        force, velocity, state = self._compute(landmark_positions_3d, timestamp)
        return JumpData(force=force, velocity=velocity, jump_state=state, timestamp=timestamp)

    def _compute(self, landmark_positions_3d, current_time):
        current_position = np.mean(landmark_positions_3d[:2, 1])
        ground = np.mean(landmark_positions_3d[2:, 1])

        if self.initial_ground is None:
            self.initial_ground = ground

        ground_change_percentage = abs(ground - self.initial_ground) / self.initial_ground
        if ground_change_percentage > 0.1:
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

def main(video_path=None):
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    tracker = JumpForceVelocityTracker(mass=70)  # Масса в килограммах
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Использование системного времени
        timestamp = time.time() - start_time

        data = tracker.update(frame, timestamp)

        if data:
            print(f"Force: {data.force:.2f}, Velocity: {data.velocity:.2f}, State: {data.jump_state.name}, Time: {data.timestamp:.2f}")

        cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()  # Для работы с камерой
    # main("path_to_video.mp4")  # Для работы с видео
