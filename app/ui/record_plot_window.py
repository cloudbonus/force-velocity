import cv2
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QImage, QPixmap, QIcon

from app.tracking.jump_tracker import JumpForceVelocityTracker, JumpData, JumpState
from app.tracking.tracking_worker import TrackingWorker
from app.utils.mlp_canvas import MplCanvas
from app.utils.video_source import VideoSource


class PlotWindow(QtWidgets.QMainWindow):
    update_video_signal = QtCore.pyqtSignal(QPixmap)
    return_to_main_signal = QtCore.pyqtSignal()

    def __init__(self, mass, video_path, model_path):
        super().__init__()
        self.setWindowTitle("Force-Velocity Profiling")
        self.setFixedSize(1280, 720)
        self.setWindowIcon(QIcon('resources/logo.png'))

        self.main_layout = QtWidgets.QVBoxLayout()

        self.placeholder_label = QtWidgets.QLabel("Загрузка видео...")
        self.placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setMinimumSize(400, 200)
        self.main_layout.addWidget(self.placeholder_label)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas.setVisible(False)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(400, 200)
        self.video_label.setVisible(False)

        self.status_label = QtWidgets.QLabel("Статус: Загрузка и обработка данных...")
        self.main_layout.addWidget(self.status_label)

        self.back_button = QtWidgets.QPushButton("Вернуться на главный экран")
        self.back_button.clicked.connect(self.return_to_main)
        self.main_layout.addWidget(self.back_button)

        central_widget = QtWidgets.QWidget(self)
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        self.tracker = JumpForceVelocityTracker(mass, video_path, model_path)
        self.worker = TrackingWorker(self.tracker)
        self.worker.data_ready.connect(self.on_new_data)
        self.worker.status_update.connect(self.update_status)
        self.worker.finished.connect(self.on_processing_finished)

        self.segments = []
        self.current_segment = {JumpState.TAKEOFF: [], JumpState.LANDING: []}
        self.skip_next = False
        self.plot_updated = False

        self.video_source = VideoSource(video_path)
        self.video_timer = QtCore.QTimer(self)
        self.video_timer.timeout.connect(self.update_video_and_plot)

        self.worker.start()

    def on_new_data(self, data: JumpData):
        if data.jump_state == JumpState.TRANSITION:
            if self.current_segment[JumpState.TAKEOFF] or self.current_segment[JumpState.LANDING]:
                self.segments.append(self.current_segment)
                self.current_segment = {JumpState.TAKEOFF: [], JumpState.LANDING: []}
            self.skip_next = True
            return

        if self.skip_next:
            self.skip_next = False
            return

        if data.jump_state == JumpState.TAKEOFF:
            self.current_segment[JumpState.TAKEOFF].append(data)
        elif data.jump_state == JumpState.LANDING:
            self.current_segment[JumpState.LANDING].append(data)

    def on_processing_finished(self):
        self.status_label.setText("Обработка завершена. Начало синхронного воспроизведения.")
        self.video_timer.start(30)  # 30 FPS

    def update_video_and_plot(self):
        try:
            frame = next(self.video_source.stream_bgr())
            rgb_frame = cv2.cvtColor(frame.data, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_frame.shape
            bytes_per_line = channel * width
            q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.width(),
                self.video_label.height(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)

            if not self.plot_updated:
                self.main_layout.replaceWidget(self.placeholder_label, self.video_label)
                self.video_label.setVisible(True)
                self.main_layout.insertWidget(0, self.canvas)  # Вставляем график в начало
                self.canvas.setVisible(True)
                self.plot_updated = True

            current_frame_idx = frame.idx
            current_frame_time = frame.time
            if current_frame_idx % 5 == 0:
                segments = [
                    segment for segment in self.segments
                    if segment[JumpState.LANDING][-1].timestamp < current_frame_time
                ]
                self.canvas.update_plot(segments)
        except StopIteration:
            self.video_timer.stop()
            self.status_label.setText("Воспроизведение завершено.")

    def return_to_main(self):
        self.video_source.close()
        self.worker.stop()
        self.close()
        self.return_to_main_signal.emit()

    def update_status(self, status):
        self.status_label.setText(f"Статус: {status}")
