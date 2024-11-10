from argparse import ArgumentParser
from typing import Final
import logging
import requests
from pathlib import Path
import os

class PoseUtils:
    DESCRIPTION: Final = "Uses the MediaPipe Pose solution to track a human pose in video."
    MODEL_URL_TEMPLATE: Final = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{model_name}/float16/latest/pose_landmarker_{model_name}.task"
    )

    def __init__(self):
        self.project_dir = Path(os.path.dirname(__file__)).parent
        self.dataset_dir = self.project_dir / "dataset" / "pose_movement"
        self.model_dir = self.project_dir / "model" / "pose_movement"

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description=self.DESCRIPTION)
        parser.add_argument(
            "--video",
            type=str,
            default="jump",
            choices=["jump"],
            help="The example video to run on.",
        )
        parser.add_argument(
            "--dataset-dir",
            type=Path,
            default=self.dataset_dir,
            help="Directory to save example videos to.",
        )
        parser.add_argument(
            "--video-path",
            type=str,
            default="",
            help="Full path to video to run on. Overrides `--video`.",
        )
        parser.add_argument(
            "--no-segment",
            action="store_true",
            help="Don't run person segmentation.",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="heavy",
            choices=["lite", "full", "heavy"],
            help="The mediapipe model to use (see https://developers.google.com/mediapipe/solutions/vision/pose_landmarker).",
        )
        parser.add_argument(
            "--model-dir",
            type=Path,
            default=self.model_dir,
            help="Directory to save downloaded model to.",
        )
        parser.add_argument(
            "--model-path",
            type=str,
            default="",
            help="Full path of mediapipe model. Overrides `--model`.",
        )
        parser.add_argument(
            "--max-frame",
            type=int,
            help="Stop after processing this many frames. If not specified, will run until interrupted.",
        )
        return parser

    # def get_video_path(self, dataset_dir: Path, video_name: str) -> str:
    #     video_file_name = f"{video_name}.mp4"
    #     destination_path = dataset_dir / video_file_name
    #     return str(destination_path)

    # def get_downloaded_model_path(model_dir: Path, model_name: str) -> str:
    #     model_file_name = f"{model_name}.task"
    #     destination_path = model_dir / model_file_name
    #     if destination_path.exists():
    #         logging.info("%s already exists. No need to download", destination_path)
    #         return str(destination_path)
    #
    #     model_url = MODEL_URL_TEMPLATE.format(model_name=model_name)
    #     logging.info("Downloading model from %s to %s", model_url, destination_path)
    #     download(model_url, destination_path)
    #     return str(destination_path)
    #
    def get_video_path(self, video_name: str) -> str:
        video_file_name = f"{video_name}.mp4"
        destination_path = self.dataset_dir / video_file_name
        return str(destination_path)

    def get_downloaded_model_path(self, model_name: str) -> str:
        model_file_name = f"{model_name}.task"
        destination_path = self.model_dir / model_file_name
        if destination_path.exists():
            logging.info("%s already exists. No need to download", destination_path)
            return str(destination_path)

        model_url = self.MODEL_URL_TEMPLATE.format(model_name=model_name)
        logging.info("Downloading model from %s to %s", model_url, destination_path)
        self.download(model_url, destination_path)
        return str(destination_path)

    def download(self, url: str, destination_path: Path) -> None:
        os.makedirs(destination_path.parent, exist_ok=True)
        with requests.get(url, stream=True) as req:
            req.raise_for_status()
            with open(destination_path, "wb") as f:
                for chunk in req.iter_content(chunk_size=8192):
                    f.write(chunk)