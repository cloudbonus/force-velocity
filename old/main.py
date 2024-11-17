#!/usr/bin/env python3

from __future__ import annotations

import logging
from contextlib import closing
from typing import Any

import cv2
import mediapipe as mp
import mediapipe.python.solutions.pose as mp_pose
import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from old import utils
from app.video_source import VideoSource

mass = 70.0

previous_position = None
previous_velocity = 0
previous_time = None
previous_force = 0


def track_pose(video_path: str, model_path: str, *, max_frame_count: int | None) -> None:
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=model_path,
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_segmentation_masks=True,
    )

    rr.log("description", rr.TextDocument(utils.DESCRIPTION, media_type=rr.MediaType.MARKDOWN), static=True)

    rr.log("curves/parabola",
           rr.SeriesLine(name="Eccentric Contraction / Concentric Contraction"),
           static=True)

    rr.log(
        "/",
        rr.AnnotationContext(
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=1),
                keypoint_connections=mp_pose.POSE_CONNECTIONS,
            )
        ),
        static=True,
    )

    rr.log(
        "video/mask",
        rr.AnnotationContext([
            rr.AnnotationInfo(id=0, label="Background"),
            rr.AnnotationInfo(id=1, label="Person", color=(0, 0, 0)),
        ]),
        static=True,
    )

    rr.log("person", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    with closing(VideoSource(video_path)) as video_source:

        for idx, bgr_frame in enumerate(video_source.stream_bgr()):

            if max_frame_count is not None and idx >= max_frame_count:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bgr_frame.data)
            rgb = cv2.cvtColor(bgr_frame.data, cv2.COLOR_BGR2RGB)
            rr.set_time_seconds("time", bgr_frame.time)
            rr.set_time_sequence("frame_idx", bgr_frame.idx)

            results = pose_landmarker.detect_for_video(mp_image, int(bgr_frame.time * 1000))
            h, w, _ = rgb.shape

            landmark_positions_2d = read_landmark_positions_2d(results, w, h)
            rr.log("video/rgb", rr.Image(rgb).compress(jpeg_quality=75))
            if landmark_positions_2d is not None:
                rr.log(
                    "video/pose/points",
                    rr.Points2D(landmark_positions_2d, class_ids=1, keypoint_ids=mp_pose.PoseLandmark),
                )

            landmark_positions_3d = read_landmark_positions_3d(results)
            idx = bgr_frame.idx
            current_time = bgr_frame.time

            force, velocity = compute_force_velocity(landmark_positions_3d, current_time, mass)

            rr.set_time_sequence("frame_nr", int(idx))
            color = [255, 255, 0]
            width = 5
            if velocity <= 0:
                color = [0, 255, 0]
            elif velocity > 0:
                color = [255, 0, 0]
            rr.log(
                "curves/parabola",
                rr.Scalar(force),
                rr.SeriesLine(width=width, color=color),
            )

            if results.segmentation_masks is not None:
                segmentation_mask = results.segmentation_masks[0].numpy_view()
                binary_segmentation_mask = segmentation_mask > 0.5
                rr.log("video/mask", rr.SegmentationImage(binary_segmentation_mask.astype(np.uint8)))


def read_landmark_positions_2d(
        results: Any,
        image_width: int,
        image_height: int,
) -> npt.NDArray[np.float32] | None:
    if results.pose_landmarks is None or len(results.pose_landmarks) == 0:
        return None
    else:
        pose_landmarks = results.pose_landmarks[0]
        normalized_landmarks = [pose_landmarks[lm] for lm in mp_pose.PoseLandmark]
        return np.array([(image_width * lm.x, image_height * lm.y) for lm in normalized_landmarks])


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

    if landmark_positions_3d is None and previous_time is not None:
        estimated_velocity = previous_velocity
        estimated_force = previous_force

        previous_time = current_time
        return estimated_force, estimated_velocity

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
    force = mass * acceleration

    previous_position = current_position
    previous_velocity = current_velocity
    previous_time = current_time
    previous_force = force

    return force, current_velocity


def main() -> None:
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel("INFO")

    parser = utils.get_parser()

    rr.script_add_args(parser)

    args = parser.parse_args()

    rr.script_setup(
        args,
        "rerun_force_velocity",
        default_blueprint=rrb.Horizontal(
            rrb.Grid(
                rrb.Spatial2DView(origin="video", name="Result"),
                rrb.Spatial3DView(origin="person", name="3D pose"),

                rrb.TimeSeriesView(origin="/curves", name="Force-velocity"),
                rrb.Spatial2DView(origin="video/rgb", name="Raw video"),
                rrb.TextDocumentView(origin="description", name="Description")
            )
        ),
    )

    video_path = args.video_path
    if not video_path:
        video_path = utils.get_video_path(args.video)

    model_path = args.model_path
    if not args.model_path:
        model_path = utils.get_downloaded_model_path(args.model)

    track_pose(video_path, model_path, max_frame_count=args.max_frame)

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
