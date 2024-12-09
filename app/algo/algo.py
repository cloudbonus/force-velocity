from contextlib import closing
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict

import mediapipe as mp
import numpy as np

from app.utils.video_source import VideoSource
import matplotlib.pyplot as plt


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
    else:
        pose_landmarks = results.pose_landmarks[0]
        landmarks = [pose_landmarks[lm] for lm in [23, 24, 31, 32]]
        return np.array([(lm.x, lm.y, lm.z) for lm in landmarks])

def create_plot(jumpdata_segments: List[Dict[JumpState, List[JumpData]]]):
    plt.figure(figsize=(12, 8))

    state_colors = {
        JumpState.TAKEOFF: "green",
        JumpState.LANDING: "red"
    }

    group_lines = {}
    for segment_index, segment in enumerate(jumpdata_segments):
        takeoff_x = []
        takeoff_y = []
        landing_x = []
        landing_y = []

        for data in segment.get(JumpState.TAKEOFF, []):
            takeoff_x.append(data.velocity)
            takeoff_y.append(data.force)

        for data in segment.get(JumpState.LANDING, []):
            landing_x.append(data.velocity)
            landing_y.append(data.force)

        takeoff_line, = plt.plot(
            takeoff_x, takeoff_y, color=state_colors[JumpState.TAKEOFF],
            label=f"Segment {segment_index + 1} Takeoff"
        )
        landing_line, = plt.plot(
            landing_x, landing_y, color=state_colors[JumpState.LANDING],
            label=f"Segment {segment_index + 1} Landing"
        )

        group_lines[f"Segment {segment_index + 1}"] = [takeoff_line, landing_line]

    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Force (N)")
    plt.title("Force-Velocity Profile for Jump Segments")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)

    def on_legend_click(event):
        for legline, segment_name in zip(plt.gca().get_legend().get_lines(), group_lines.keys()):
            if legline == event.artist:
                for line in group_lines[segment_name]:
                    visible = not line.get_visible()
                    line.set_visible(visible)
                legline.set_alpha(1.0 if visible else 0.2)
                plt.draw()

    legend = plt.gca().get_legend()
    legend.set_picker(True)
    for legline in legend.get_lines():
        legline.set_picker(5)

    plt.gcf().canvas.mpl_connect('pick_event', on_legend_click)
    plt.tight_layout()
    plt.show()

class ForceVelocityTracker:
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

    def update(self) -> List[Dict[JumpState, List[JumpData]]]:
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
                force, velocity, state = self._compute_force_velocity(landmark_positions_3d, current_time)

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

    def _compute_force_velocity(self, landmark_positions_3d, current_time):
        current_position = (landmark_positions_3d[0][1] + landmark_positions_3d[1][1]) / 2

        ground = (landmark_positions_3d[2][1] + landmark_positions_3d[3][1]) / 2

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


        current_velocity = delta_y / delta_t

        acceleration = (current_velocity - self.previous_velocity) / delta_t

        if abs(current_velocity) > 1e-5:  # Избегаем деления на ноль
            force = self.mass * abs(acceleration) / abs(current_velocity)
        else:
            force = self.mass * abs(acceleration)

        error_margin = 0.001 * abs(self.previous_position)
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
    tracker = ForceVelocityTracker(
        mass=70,
        video_path="../../dataset/pose_movement/jump.mp4",
        model_path="../../model/pose_movement/heavy.task",
    )
    data = tracker.update()
    print(data)

    # with open('jump_data.pkl', 'wb') as file:
    #     pickle.dump(data, file)
    # with open('jump_data.pkl', 'rb') as file:
    #     data = pickle.load(file)

    create_plot(data)