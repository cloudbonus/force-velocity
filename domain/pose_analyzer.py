import pickle
import logging
import sys
import numpy as np
import mediapipe as mp
import numpy.typing as npt
import utils
from typing import Any
from domain.video_source import VideoFrame


class VideoProcessor:
    def __init__(self, queue_name: str, model_path: str, mass: float = 70.0) -> None:
        self.queue_name = queue_name
        self.model_path = model_path
        self.mass = mass
        self.connection = None
        self.channel = None
        self.previous_position = None
        self.previous_velocity = 0
        self.previous_time = None
        self.previous_force = 0

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_segmentation_masks=True,
        )

        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def connect_rabbitmq(self) -> None:
        try:
            self.connection = utils.create_connection()
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.queue_name)
            self.channel.basic_consume(queue=self.queue_name, on_message_callback=self.callback, auto_ack=True)
        except Exception as e:
            logging.error(f"Failed to connect to RabbitMQ: {e}")
            sys.exit(1)

    def callback(self, ch, method, properties, body):
        frame: VideoFrame = pickle.loads(body)
        bgr_frame = frame.data
        current_time = frame.time
        idx = frame.idx

        self.process_frame(bgr_frame, current_time, idx)

    def process_frame(self, bgr_frame: np.ndarray, current_time: float, idx: int) -> None:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bgr_frame)

        results = self.pose_landmarker.detect_for_video(mp_image, int(current_time * 1000))
        landmark_positions_3d = self.read_landmark_positions_3d(results)
        force, velocity = self.compute_force_velocity(landmark_positions_3d, current_time)

    def read_landmark_positions_3d(self, results: Any) -> npt.NDArray[np.float32] | None:
        if results.pose_landmarks is None:
            return None
        pose_landmarks = results.pose_landmarks[0]
        landmarks = [pose_landmarks[lm] for lm in [23, 24, 29, 30]]
        return np.array([(lm.x, lm.y, lm.z) for lm in landmarks])

    def compute_force_velocity(self, landmark_positions_3d, current_time):
        if landmark_positions_3d is None:
            return 0, 0

        current_position = (landmark_positions_3d[0][1] + landmark_positions_3d[1][1]) / 2
        if self.previous_position is None:
            self.previous_position = current_position
            self.previous_time = current_time
            self.previous_velocity = 0
            self.previous_force = 0
            return 0, 0

        delta_y = current_position - self.previous_position
        delta_t = current_time - self.previous_time

        if delta_t == 0:
            return self.previous_force, self.previous_velocity

        current_velocity = delta_y / delta_t
        acceleration = (current_velocity - self.previous_velocity) / delta_t
        force = self.mass * acceleration

        self.previous_position = current_position
        self.previous_velocity = current_velocity
        self.previous_time = current_time
        self.previous_force = force

        return force, current_velocity

    def start_processing(self) -> None:
        self.connect_rabbitmq()
        logging.info("Started processing video")
        self.channel.start_consuming()

    def close_rabbitmq(self) -> None:
        if self.connection:
            self.connection.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    queue_name = 'pose_analysis_queue'
    parser = utils.get_parser()

    args = parser.parse_args()

    model_path = args.model_path
    if not args.model_path:
        model_path = utils.get_downloaded_model_path(args.model)

    video_processor = VideoProcessor(queue_name, model_path)
    video_processor.connect_rabbitmq()
    video_processor.start_processing()
    video_processor.close_rabbitmq()