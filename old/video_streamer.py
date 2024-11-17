import pickle
import logging
import sys
from app.video_source import VideoSource
import utils
from contextlib import closing

class VideoSender:
    def __init__(self, video_path: str, queue_name: str) -> None:
        self.video_path = video_path
        self.queue_name = queue_name
        self.connection = None
        self.channel = None

    def connect_rabbitmq(self) -> None:
        try:
            self.connection = utils.create_connection()
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.queue_name)
        except Exception as e:
            logging.error(f"Failed to connect to RabbitMQ: {e}")
            sys.exit(1)

    def send_frame(self, frame: bytes) -> None:
        self.channel.basic_publish(exchange='',
                                   routing_key=self.queue_name,
                                   body=frame)
        logging.info("Sent frame to RabbitMQ")

    def process_video(self) -> None:
        with closing(VideoSource(video_path)) as video_source:
            for idx, bgr_frame in enumerate(video_source.stream_bgr()):
                serialized_frame = pickle.dumps(bgr_frame)
                self.send_frame(serialized_frame)
                logging.info(f"Sent frame {idx}")

    def close_rabbitmq(self) -> None:
        if self.connection:
            self.connection.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    queue_name = 'pose_analysis_queue'
    parser = utils.get_parser()

    args = parser.parse_args()

    video_path = args.video_path
    if not video_path:
        video_path = utils.get_video_path(args.video)

    video_sender = VideoSender(video_path, queue_name)
    video_sender.connect_rabbitmq()
    video_sender.process_video()
    video_sender.close_rabbitmq()
