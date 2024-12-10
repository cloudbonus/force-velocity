from PyQt6 import QtWidgets, QtCore, QtGui
import cv2
import time

from PyQt6.QtGui import QIcon

from jump_tracker import JumpData, JumpState, CameraJumpForceVelocityTracker
from mlp_canvas import MplCanvas

class CameraPlotWindow(QtWidgets.QMainWindow):
    return_to_main_signal = QtCore.pyqtSignal()

    def __init__(self, mass):
        super().__init__()
        self.setWindowTitle("Force-Velocity Profiling")
        self.setGeometry(100, 100, 1000, 800)
        self.setWindowIcon(QIcon('resources/logo.png'))

        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.graph_canvas = MplCanvas(self)
        self.graph_canvas.setVisible(False)  # Hide until countdown ends
        self.layout.addWidget(self.graph_canvas)

        self.timer_label = QtWidgets.QLabel("Подготовьтесь... 5")
        self.timer_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.layout.addWidget(self.timer_label)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.video_label)

        self.status_label = QtWidgets.QLabel("Status: Ready")
        self.layout.addWidget(self.status_label)

        self.return_button = QtWidgets.QPushButton("Вернуться на главный экран")
        self.return_button.clicked.connect(self.return_to_main)
        self.layout.addWidget(self.return_button)

        self.cap = cv2.VideoCapture(0)
        self.tracker = CameraJumpForceVelocityTracker(mass=mass)

        self.segments = []
        self.current_segment = {JumpState.TAKEOFF: [], JumpState.LANDING: []}
        self.skip_next = False
        self.countdown = 5

        self.countdown_timer = QtCore.QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.graph_update_timer = QtCore.QTimer(self)
        self.graph_update_timer.timeout.connect(self.update_graph)

        self.start_time = None

    def update_countdown(self):
        self.countdown -= 1
        self.timer_label.setText(f"Подготовьтесь... {self.countdown}")
        if self.countdown == 0:
            self.countdown_timer.stop()
            self.timer_label.hide()
            self.graph_canvas.setVisible(True)
            self.start_time = time.time()
            self.timer.start(30)
            self.graph_update_timer.start(10000)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("Status: Camera not available")
            return

        timestamp = time.time() - self.start_time
        data = self.tracker.update_for_camera(frame, timestamp)

        if data:
            self.on_new_data(data)
            self.status_label.setText(
                f"Force: {data.force:.2f}, Velocity: {data.velocity:.2f}, State: {data.jump_state.name}"
            )
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        q_image = QtGui.QImage(rgb_image.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

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

    def update_graph(self):
        segments_to_display = self.segments.copy()
        self.graph_canvas.update_plot(segments_to_display)

    def return_to_main(self):
        self.timer.stop()
        self.graph_update_timer.stop()
        self.cap.release()
        self.close()
        self.return_to_main_signal.emit()