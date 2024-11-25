import csv

import mediapipe as mp
import numpy as np
import scipy.signal

from app.utils.video_source import VideoSource


class ForceVelocityTracker:
    def __init__(self, mass, video_path, model_path):
        self.mass = mass
        self.video_path = video_path
        self.model_path = model_path

        self.previous_position = None
        self.previous_velocity = 0
        self.previous_time = None
        self.previous_force = 0

        self.forces = []
        self.velocities = []

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

        landmark_positions_3d = self._read_landmark_positions_3d(results)
        if landmark_positions_3d is None:
            return None

        current_time = frame.time
        force, velocity = self._compute_force_velocity(landmark_positions_3d, current_time)

        self.forces.append(force)
        self.velocities.append(velocity)

        return self._filter_data()

    def _filter_data(self):
        kernel_size = 3
        if len(self.forces) >= 3:
            filtered_forces = scipy.signal.medfilt(self.forces, kernel_size=kernel_size)
            filtered_velocities = scipy.signal.medfilt(self.velocities, kernel_size=kernel_size)
        else:
            filtered_forces = self.forces
            filtered_velocities = self.velocities

        return filtered_forces, filtered_velocities

    def _read_landmark_positions_3d(self, results):
        if results.pose_landmarks is None or len(results.pose_landmarks) == 0:
            return None
        else:
            pose_landmarks = results.pose_landmarks[0]
            landmarks = [pose_landmarks[lm] for lm in [23, 24, 31, 32]]
            return np.array([(lm.x, lm.y, lm.z) for lm in landmarks])

    def _compute_force_velocity(self, landmark_positions_3d, current_time):
        current_position = (landmark_positions_3d[0][1] + landmark_positions_3d[1][1]) / 2

        ground = (landmark_positions_3d[2][1] + landmark_positions_3d[3][1]) / 2  # Носки человека

        # Установить базовый уровень ground, если это первый кадр
        if not hasattr(self, "initial_ground"):
            self.initial_ground = ground

        # Пропустить кадры, где изменение уровня ground > 10% от базового значения
        ground_change_percentage = abs(ground - self.initial_ground) / self.initial_ground
        if ground_change_percentage > 0.05:  # Пропуск, если изменение больше 10%
            return self.previous_force, self.previous_velocity

        if self.previous_position is None:
            self.previous_position = current_position
            self.previous_time = current_time
            return 0, 0

        delta_y = current_position - self.previous_position
        delta_t = current_time - self.previous_time

        if delta_t == 0:
            return self.previous_force, self.previous_velocity

        current_velocity = delta_y / delta_t
        acceleration = (current_velocity - self.previous_velocity) / delta_t
        force = self.mass * abs(acceleration)

        self.previous_position = current_position
        self.previous_velocity = current_velocity
        self.previous_time = current_time
        self.previous_force = force

        return force, current_velocity

    def save_to_csv(self, filename='force_velocity_data.csv'):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time (s)', 'Force (N)', 'Velocity (m/s)'])

            for i, (force, velocity) in enumerate(zip(self.forces, self.velocities)):
                writer.writerow([i, force, velocity])
