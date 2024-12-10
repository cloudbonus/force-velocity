from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap, QIcon

from app.tracking.jump_tracker import JumpForceVelocityTracker, JumpData, JumpState
from app.utils.camera_tracking_worker import CameraTrackingWorker
from app.utils.mlp_canvas import MplCanvas


class CameraPlotWindow(QtWidgets.QMainWindow):
    update_video_signal = QtCore.pyqtSignal(QPixmap)
    return_to_main_signal = QtCore.pyqtSignal()

    def __init__(self, mass, model_path):
        super().__init__()
        video_path = "0"

        self.setWindowTitle("Force-Velocity Profiling")
        self.setFixedSize(1280, 720)
        self.setWindowIcon(QIcon('resources/logo.png'))

        self.main_layout = QtWidgets.QVBoxLayout()

        self.placeholder_label = QtWidgets.QLabel("Подготовьтесь...")
        self.placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setMinimumSize(400, 200)
        self.main_layout.addWidget(self.placeholder_label)

        self.timer_label = QtWidgets.QLabel("5")
        self.timer_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 48px; font-weight: bold;")
        self.main_layout.addWidget(self.timer_label)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas.setVisible(False)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(400, 200)
        self.video_label.setVisible(False)

        self.status_label = QtWidgets.QLabel("")
        self.main_layout.addWidget(self.status_label)

        self.back_button = QtWidgets.QPushButton("Вернуться на главный экран")
        self.back_button.clicked.connect(self.return_to_main)
        self.main_layout.addWidget(self.back_button)

        central_widget = QtWidgets.QWidget(self)
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        self.tracker = JumpForceVelocityTracker(mass, video_path, model_path)
        self.worker = CameraTrackingWorker(self.tracker)
        self.worker.data_ready.connect(self.on_new_data)
        self.worker.status_update.connect(self.update_status)
        self.worker.finished.connect(self.on_processing_finished)

        self.segments = []
        self.current_segment = {JumpState.TAKEOFF: [], JumpState.LANDING: []}
        self.skip_next = False
        self.plot_updated = False

        self.start_timer()

    def start_timer(self):
        self.countdown = 5  # 5 seconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)  # Update every second

    def update_timer(self):
        self.countdown -= 1
        if self.countdown > 0:
            self.timer_label.setText(str(self.countdown))
        else:
            self.timer.stop()
            self.timer_label.setVisible(False)
            self.worker.start()

    def on_new_data(self, data: JumpData):
        if not self.plot_updated:
            self.main_layout.replaceWidget(self.placeholder_label, self.video_label)
            self.main_layout.insertWidget(0, self.canvas)
            self.canvas.setVisible(True)
            self.plot_updated = True

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
        pass  # 30 FPS

    def return_to_main(self):
        self.worker.stop()
        self.close()
        self.return_to_main_signal.emit()

    def update_status(self, status):
        self.status_label.setText(f"Статус: {status}")
