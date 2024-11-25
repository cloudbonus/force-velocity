from PyQt6 import QtWidgets, QtGui, QtCore
from app.utils.mlp_canvas import MplCanvas
from app.tracking.tracker import ForceVelocityTracker
from app.tracking.tracking_worker import TrackingWorker
from app.utils.video_player import VideoPlayer


class PlotWindow(QtWidgets.QMainWindow):
    update_video_signal = QtCore.pyqtSignal(QtGui.QPixmap)

    return_to_main_signal = QtCore.pyqtSignal()

    def __init__(self, mass, video_path, model_path):
        super().__init__()
        self.setWindowTitle("Force-Velocity Profiling")
        self.setFixedSize(800, 600)
        self.setWindowIcon(QtGui.QIcon('resources/logo.png'))

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
        self.status_label = QtWidgets.QLabel("Статус: Ожидание...")
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
        self.worker.start()

        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self.update_plot_periodically)
        self.plot_timer.start(1000)

        self.forces = []
        self.velocities = []

        # Video handling
        self.video_player = VideoPlayer(video_path)
        self.video_player.frame_ready.connect(self.update_video_label)
        self.video_player.video_finished.connect(self.on_video_finished)
        self.video_player.start()

    def on_new_data(self, forces, velocities):
        self.forces = forces
        self.velocities = velocities

    def update_plot_periodically(self):
        if self.forces and self.velocities:
            self.canvas.update_plot(self.forces, self.velocities)

    def update_status(self, status):
        self.status_label.setText(f"Статус: {status}")

    def update_video_label(self, pixmap):
        # Масштабируем видео, чтобы оно помещалось в QLabel, сохраняя пропорции
        scaled_pixmap = pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def on_video_finished(self):
        self.status_label.setText("Видео завершено")

    def return_to_main(self):
        self.video_player.stop()
        self.tracker.save_to_csv()
        self.worker.stop()
        self.close()
        self.return_to_main_signal.emit()

    def closeEvent(self, event):
        self.video_player.stop()
        self.worker.stop()
        super().closeEvent(event)
