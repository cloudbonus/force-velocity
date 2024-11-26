import cv2
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QImage, QPixmap, QIcon
from app.utils.mlp_canvas import MplCanvas
from app.tracking.tracker import ForceVelocityTracker
from app.tracking.tracking_worker import TrackingWorker
from app.utils.video_source import VideoSource


class PlotWindow(QtWidgets.QMainWindow):
    update_video_signal = QtCore.pyqtSignal(QPixmap)
    return_to_main_signal = QtCore.pyqtSignal()

    def __init__(self, mass, video_path, model_path):
        super().__init__()
        self.setWindowTitle("Force-Velocity Profiling")
        self.setFixedSize(800, 600)
        self.setWindowIcon(QIcon('resources/logo.png'))

        self.main_layout = QtWidgets.QVBoxLayout()

        # Canvas for plot
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.main_layout.addWidget(self.canvas)

        # Video widget
        self.video_label = QtWidgets.QLabel("Загрузка видео...")
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(400, 200)  # Минимальный размер для видео
        self.main_layout.addWidget(self.video_label)

        # Status label
        self.status_label = QtWidgets.QLabel("Статус: Загрузка и обработка данных...")
        self.main_layout.addWidget(self.status_label)

        # Back button
        self.back_button = QtWidgets.QPushButton("Вернуться на главную")
        self.back_button.clicked.connect(self.return_to_main)
        self.main_layout.addWidget(self.back_button)

        # Central widget
        central_widget = QtWidgets.QWidget(self)
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        # Tracking and video handling
        self.tracker = ForceVelocityTracker(mass, video_path, model_path)
        self.worker = TrackingWorker(self.tracker)
        self.worker.data_ready.connect(self.on_new_data)
        self.worker.status_update.connect(self.update_status)
        self.worker.finished.connect(self.on_processing_finished)

        self.forces = []
        self.velocities = []

        # Video handling
        self.video_source = VideoSource(video_path)
        self.video_timer = QtCore.QTimer(self)
        self.video_timer.timeout.connect(self.update_video_and_plot)

        # Start preprocessing
        self.worker.start()

    def on_new_data(self, forces, velocities):
        self.forces = forces
        self.velocities = velocities

    def on_processing_finished(self):
        self.status_label.setText("Обработка завершена. Начало синхронного воспроизведения.")
        self.video_timer.start(30)  # Ожидаем около 30 FPS

    def update_video_and_plot(self):
        try:
            # Обновление видео
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

            # Обновление графика
            current_frame_idx = frame.idx
            if current_frame_idx % 5 == 0 and current_frame_idx < len(self.forces):
                current_forces = self.forces[:current_frame_idx]
                current_velocities = self.velocities[:current_frame_idx]
                self.canvas.update_plot(current_forces, current_velocities)
        except StopIteration:
            # Завершение воспроизведения
            self.video_timer.stop()
            self.status_label.setText("Воспроизведение завершено.")

    def return_to_main(self):
        self.video_source.close()
        self.tracker.save_to_csv()
        self.worker.stop()
        self.close()
        self.return_to_main_signal.emit()

    def closeEvent(self, event):
        self.video_source.close()
        self.worker.stop()
        super().closeEvent(event)

    def update_plot_periodically(self):
        if self.forces and self.velocities:
            self.canvas.update_plot(self.forces, self.velocities)

    def update_status(self, status):
        self.status_label.setText(f"Статус: {status}")

    def update_video_frame(self):
        try:
            frame = next(self.video_source.stream_bgr())
            # Преобразование кадра в формат QPixmap
            rgb_frame = cv2.cvtColor(frame.data, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_frame.shape
            bytes_per_line = channel * width
            q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # Масштабируем видео, чтобы оно помещалось в QLabel, сохраняя пропорции
            scaled_pixmap = pixmap.scaled(
                self.video_label.width(),
                self.video_label.height(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
        except StopIteration:
            # Если кадры закончились, прекращаем таймер
            self.video_timer.stop()
            self.status_label.setText("Видео завершено, продолжаем обработку...")
            self.worker.set_video_finished()

