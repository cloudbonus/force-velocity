import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap


class VideoPlayer(QThread):
    frame_ready = pyqtSignal(QPixmap)  # Сигнал для передачи текущего кадра
    video_finished = pyqtSignal()  # Сигнал, который отправляется по завершении видео

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.capture = None
        self.running = False

    def run(self):
        """Основной поток воспроизведения видео."""
        self.running = True
        self.capture = cv2.VideoCapture(self.video_path)

        if not self.capture.isOpened():
            print("Ошибка: Не удалось открыть видео.")
            return

        fps = self.capture.get(cv2.CAP_PROP_FPS)
        frame_delay = 1 / fps if fps > 0 else 0.033  # Задержка между кадрами

        while self.running and self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                break

            # Преобразование кадра в формат QPixmap
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_frame.shape
            bytes_per_line = channel * width
            q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # Отправляем сигнал с новым кадром
            self.frame_ready.emit(pixmap)

            # Задержка для синхронизации с FPS
            self.msleep(int(frame_delay * 1000))

        self.capture.release()
        self.video_finished.emit()  # Сигнал по завершении видео

    def stop(self):
        """Остановка воспроизведения видео."""
        self.running = False
        self.wait()