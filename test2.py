#!/usr/bin/env python3

from __future__ import annotations

from contextlib import closing
from typing import Any
import scipy.signal
import mediapipe as mp
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from domain import utils
from domain.video_source import VideoSource

mass = 70.0

previous_position = None
previous_velocity = 0
previous_time = None
previous_force = 0

forces = []
velocities = []

def track_pose(video_path: str, model_path: str, *, max_frame_count: int | None) -> None:
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=model_path,
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_segmentation_masks=True,
    )

    pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    with closing(VideoSource(video_path)) as video_source:

        _, ax = plt.subplots()
        ax.set_ylim(-1000, 1000)
        ax.set_xlim(-2, 2)
        plt.xlabel("V")
        plt.ylabel('F')
        plt.title("Force-Velocity")
        plt.grid()
        plt.legend()

        def update_plot():
            # Clear the axis to remove old plots
            ax.cla()

            if len(forces) > 3:
                filtered_forces = scipy.signal.medfilt(forces, kernel_size=5)
                filtered_velocities = scipy.signal.medfilt(velocities, kernel_size=5)
            else:
                filtered_forces = forces
                filtered_velocities = velocities

                # Re-plot the force-velocity points
            ax.plot(filtered_velocities, filtered_forces, 'o', label="Force-Velocity Points", markersize=3)

            # If there are enough data points, fit a polynomial and plot it
            if len(forces) > 2:
                try:
                    # Fit a polynomial to the force-velocity data points (degree 2 for quadratic fit)
                    poly_coeffs = np.polyfit(filtered_forces, filtered_velocities, deg=2)
                    poly_func = np.poly1d(poly_coeffs)

                    # Generate smoothed values for the fitted polynomial curve
                    force_range = np.linspace(min(filtered_forces), max(filtered_forces), 200)
                    velocity_range = poly_func(force_range)

                    # Plot the new fitted polynomial curve
                    ax.plot(velocity_range, force_range, 'r-', label="Polynomial Fit")
                except ValueError:
                    # Ignore errors if interpolation is not possible
                    pass

                
            # Set labels, title, and legend again after clearing
            ax.set_ylim(-1000, 1000)
            ax.set_xlim(-2, 2)
            plt.xlabel("V")
            plt.ylabel('F(N)')
            plt.title("Force-Velocity")
            plt.grid()
            plt.legend()

            # Update the plot
            plt.pause(0.001)

        for idx, bgr_frame in enumerate(video_source.stream_bgr()):

            if max_frame_count is not None and idx >= max_frame_count:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bgr_frame.data)

            results = pose_landmarker.detect_for_video(mp_image, int(bgr_frame.time * 1000))

            landmark_positions_3d = read_landmark_positions_3d(results)
            current_time = bgr_frame.time

            force, velocity = compute_force_velocity(landmark_positions_3d, current_time, mass)


            forces.append(force)
            velocities.append(velocity)

            update_plot()

        plt.show()

def read_landmark_positions_3d(
        results: Any,
) -> npt.NDArray[np.float32] | None:
    if results.pose_landmarks is None or len(results.pose_landmarks) == 0:
        return None
    else:
        pose_landmarks = results.pose_landmarks[0]
        landmarks = [pose_landmarks[lm] for lm in [23, 24, 29, 30]]
        return np.array([(lm.x, lm.y, lm.z) for lm in landmarks])


def compute_force_velocity(landmark_positions_3d, current_time, mass):
    global previous_position, previous_velocity, previous_time, previous_force

    current_position = (landmark_positions_3d[0][1] + landmark_positions_3d[1][1]) / 2

    if previous_position is None:
        previous_position = current_position
        previous_time = current_time
        previous_velocity = 0
        previous_force = 0
        return 0, 0

    delta_y = current_position - previous_position
    delta_t = current_time - previous_time

    if delta_t == 0:
        return previous_force, previous_velocity

    current_velocity = delta_y / delta_t
    acceleration = (current_velocity - previous_velocity) / delta_t
    force = mass * abs(acceleration)

    previous_position = current_position
    previous_velocity = current_velocity
    previous_time = current_time
    previous_force = force

    return force, current_velocity


def main() -> None:
    parser = utils.get_parser()

    args = parser.parse_args()

    video_path = args.video_path
    if not video_path:
        video_path = utils.get_video_path(args.video)

    model_path = args.model_path
    if not args.model_path:
        model_path = utils.get_downloaded_model_path(args.model)

    track_pose(video_path, model_path, max_frame_count=args.max_frame)

if __name__ == "__main__":
    main()
