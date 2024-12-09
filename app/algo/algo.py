from contextlib import closing
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict

import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from scipy.ndimage import gaussian_filter1d

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


def read_landmark_positions_3d(results):
    if results.pose_landmarks is None or len(results.pose_landmarks) == 0:
        return None
    pose_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in results.pose_landmarks[0]])
    indices = np.array([23, 24, 31, 32])
    return pose_landmarks[indices]


def plot_smoothed(data: List[Dict[JumpState, List[JumpData]]], smooth_sigma=2):
    plt.figure(figsize=(12, 8))

    takeoff_data = {}
    landing_data = {}

    for segment in data:
        for jump in segment.get(JumpState.TAKEOFF, []):
            takeoff_data.setdefault(jump.velocity, []).append(jump.force)
        for jump in segment.get(JumpState.LANDING, []):
            landing_data.setdefault(jump.velocity, []).append(jump.force)

    def aggregate_data(jump_data):
        x, y = [], []
        for velocity, forces in sorted(jump_data.items()):
            x.append(velocity)
            y.append(np.mean(forces))  # Можно заменить на np.median(forces)
        return np.array(x), np.array(y)

    takeoff_x, takeoff_y = aggregate_data(takeoff_data)
    landing_x, landing_y = aggregate_data(landing_data)

    takeoff_y_smooth = gaussian_filter1d(takeoff_y, sigma=smooth_sigma)
    landing_y_smooth = gaussian_filter1d(landing_y, sigma=smooth_sigma)

    if len(takeoff_x) > 1:
        plt.plot(takeoff_x, takeoff_y_smooth, label="Takeoff", color="green")
        plt.fill_between(
            takeoff_x, takeoff_y_smooth, color="green", alpha=0.2, label="Eccentric Phase"
        )
    if len(landing_x) > 1:
        plt.plot(landing_x, landing_y_smooth, label="Landing", color="red")
        plt.fill_between(
            landing_x, landing_y_smooth, color="red", alpha=0.2, label="Concentric Phase"
        )

    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Force (N)")
    plt.title("Smoothed Force-Velocity Profile")
    plt.axvline(color='black', linewidth=2, linestyle='--', label="Transition Point (Zero Velocity)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_segments(jumpdata_segments: List[Dict[JumpState, List[JumpData]]]):
    def on_legend_click(event):
        artist = event.artist
        for legline, segment_name in zip(legend.get_lines(), segment_lines.keys()):
            if legline == artist:
                visible = not segment_lines[segment_name][0].get_visible()
                for line in segment_lines[segment_name]:
                    line.set_visible(visible)
                legline.set_alpha(1.0 if visible else 0.2)
                plt.draw()

    def add_segment_data(segment, color, label):
        velocities = [data.velocity for data in segment]
        forces = [data.force for data in segment]
        line, = plt.plot(velocities, forces, color=color, label=label)
        return line

    plt.figure(figsize=(12, 8))

    state_colors = {
        JumpState.TAKEOFF: "green",
        JumpState.LANDING: "red"
    }

    segment_lines = {}
    legend_labels = []

    for segment_index, segment in enumerate(jumpdata_segments):
        takeoff_line = add_segment_data(
            segment.get(JumpState.TAKEOFF, []),
            state_colors[JumpState.TAKEOFF],
            f"Takeoff {segment_index + 1}"
        )
        landing_line = add_segment_data(
            segment.get(JumpState.LANDING, []),
            state_colors[JumpState.LANDING],
            f"Landing {segment_index + 1}"
        )

        segment_label = f"Segment {segment_index + 1}"
        segment_lines[segment_label] = [takeoff_line, landing_line]
        legend_labels.append(segment_label)

    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Force (N)")
    plt.title("Force-Velocity Profile for Jump Segments")
    plt.grid(True)

    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    for legline in legend.get_lines():
        legline.set_picker(5)

    plt.gcf().canvas.mpl_connect('pick_event', on_legend_click)
    plt.tight_layout()
    plt.show()


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

    def compute_force_velocity(self) -> List[Dict[JumpState, List[JumpData]]]:
        segments = []
        current_segment = {JumpState.TAKEOFF: [], JumpState.LANDING: []}

        skip_next = False

        with closing(VideoSource(self.video_path)) as video_source:
            for idx, bgr_frame in enumerate(video_source.stream_bgr()):
                if idx is None:
                    break

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bgr_frame.data)
                results = self.pose_landmarker.detect_for_video(mp_image, int(bgr_frame.time * 1000))

                landmark_positions_3d = read_landmark_positions_3d(results)
                if landmark_positions_3d is None:
                    continue

                current_time = bgr_frame.time
                force, velocity, state = self._compute(landmark_positions_3d, current_time)

                if state not in (JumpState.TAKEOFF, JumpState.LANDING, JumpState.TRANSITION):
                    continue

                if state == JumpState.TRANSITION:
                    if current_segment[JumpState.TAKEOFF] or current_segment[JumpState.LANDING]:
                        segments.append(current_segment)
                    current_segment = {JumpState.TAKEOFF: [], JumpState.LANDING: []}  # Новый сегмент
                    skip_next = True
                    continue

                if skip_next:
                    skip_next = False
                    continue

                data_entry = JumpData(force=force, velocity=velocity, jump_state=state)
                if state == JumpState.TAKEOFF:
                    current_segment[JumpState.TAKEOFF].append(data_entry)
                elif state == JumpState.LANDING:
                    current_segment[JumpState.LANDING].append(data_entry)

        if current_segment[JumpState.TAKEOFF] or current_segment[JumpState.LANDING]:
            segments.append(current_segment)

        return segments

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


if __name__ == "__main__":
    tracker = JumpForceVelocityTracker(
        mass=70,
        video_path="../../dataset/pose_movement/jump.mp4",
        model_path="../../model/pose_movement/heavy.task",
    )
    data = tracker.compute_force_velocity()
    #
    # with open('jump_data.pkl', 'wb') as file:
    #     pickle.dump(data, file)
    #
    # with open('jump_data.pkl', 'rb') as file:
    #     data = pickle.load(file)
    #     plot_smoothed(data)
    # create_plot(data)
    plot_segments(data)
    plot_smoothed(data)
# create_plot(data)
