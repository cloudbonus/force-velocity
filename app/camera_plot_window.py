from PyQt6 import QtWidgets, QtCore, QtGui
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from enum import Enum
import sys
import time
from mlp_canvas import MplCanvas
import pickle

class JumpState(Enum):
    TAKEOFF = 1
    LANDING = 2
    UNKNOWN = 3
    TRANSITION = 4


@dataclass
class JumpData:
    force: int
    velocity: int
    jump_state: JumpState
    timestamp: float


def read_landmark_positions_3d(results):
    if not results or not results.pose_landmarks:
        return None
    indices = [23, 24, 31, 32]
    pose_landmarks = np.array(
        [(lm.x, lm.y, lm.z) for i, lm in enumerate(results.pose_landmarks.landmark) if i in indices]
    )
    return pose_landmarks


class JumpForceVelocityTracker:
    def __init__(self, mass):
        self.mass = mass
        self.previous_position = None
        self.initial_ground = None
        self.previous_time = None
        self.previous_velocity = 0
        self.previous_force = 0
        self.previous_state = JumpState.UNKNOWN

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def update(self, frame, timestamp):
        mp_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(mp_image)
        landmark_positions_3d = read_landmark_positions_3d(results)
        if landmark_positions_3d is None:
            return None

        force, velocity, state = self._compute(landmark_positions_3d, timestamp)
        return JumpData(force=force, velocity=velocity, jump_state=state, timestamp=timestamp)

    def _compute(self, landmark_positions_3d, current_time):
        current_position = np.mean(landmark_positions_3d[:2, 1])
        ground = np.mean(landmark_positions_3d[2:, 1])

        if self.initial_ground is None:
            self.initial_ground = ground

        ground_change_percentage = abs(ground - self.initial_ground) / self.initial_ground
        if ground_change_percentage > 0.1:
            return self.previous_force, self.previous_velocity, JumpState.TRANSITION

        if self.previous_position is None:
            self.previous_position = current_position
            self.previous_time = current_time
            return 0, 0, JumpState.UNKNOWN

        delta_y = current_position - self.previous_position
        delta_t = current_time - self.previous_time

        if delta_t <= 0:
            return self.previous_force, self.previous_velocity, self.previous_state

        current_velocity = -delta_y / delta_t
        acceleration = (current_velocity - self.previous_velocity) / delta_t

        force = (
            self.mass * np.abs(acceleration) / np.abs(current_velocity)
            if np.abs(current_velocity) > 1e-5
            else self.mass * np.abs(acceleration)
        )

        error_margin = 0.001 * np.abs(self.previous_position)
        if current_position >= self.previous_position + error_margin:
            state = JumpState.LANDING
        elif current_position <= self.previous_position - error_margin:
            state = JumpState.TAKEOFF
        else:
            state = JumpState.UNKNOWN

        self.previous_position = current_position
        self.previous_velocity = current_velocity
        self.previous_time = current_time
        self.previous_force = force
        self.previous_state = state

        return force, current_velocity, state


class CameraPlotWindow(QtWidgets.QMainWindow):
    return_to_main_signal = QtCore.pyqtSignal()

    def __init__(self, mass):
        super().__init__()
        self.setWindowTitle("Jump Force-Velocity Tracker")
        self.setGeometry(100, 100, 1000, 800)

        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Add graph canvas at the top
        self.graph_canvas = MplCanvas(self)
        self.graph_canvas.setVisible(False)  # Hide until countdown ends
        self.layout.addWidget(self.graph_canvas)

        # Timer label for countdown
        self.timer_label = QtWidgets.QLabel("Подготовьтесь... 5")
        self.timer_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.layout.addWidget(self.timer_label)

        # Video feed
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.video_label)

        # Status label
        self.status_label = QtWidgets.QLabel("Status: Ready")
        self.layout.addWidget(self.status_label)

        # Return button
        self.return_button = QtWidgets.QPushButton("Вернуться на главный экран")
        self.return_button.clicked.connect(self.return_to_main)
        self.layout.addWidget(self.return_button)

        # Initialize video capture and tracking
        self.cap = cv2.VideoCapture(0)
        self.tracker = JumpForceVelocityTracker(mass=mass)

        # Variables for countdown and segments
        self.segments = []
        self.current_segment = {JumpState.TAKEOFF: [], JumpState.LANDING: []}
        self.skip_next = False
        self.countdown = 5

        # Timers
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
            self.graph_update_timer.start(10000)  # Update graph every 2 seconds

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("Status: Camera not available")
            return

        timestamp = time.time() - self.start_time
        data = self.tracker.update(frame, timestamp)

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
        #self.graph_canvas.update_plot(self.segments)
        with open('ui/jump_data.pkl', 'rb') as file:
            data = pickle.load(file)
            self.graph_canvas.update_plot(data)

    def return_to_main(self):
        self.timer.stop()
        self.graph_update_timer.stop()
        self.cap.release()
        self.close()
        self.return_to_main_signal.emit()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = CameraPlotWindow(mass=70)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
