from PyQt6 import QtWidgets, QtGui, QtCore

from app.tracking.tracker import ForceVelocityTracker
from app.tracking.tracking_worker import TrackingWorker
from app.utils.mlp_canvas import MplCanvas


class PlotWindow(QtWidgets.QMainWindow):
    return_to_main_signal = QtCore.pyqtSignal()

    def __init__(self, mass, video_path, model_path):
        super().__init__()
        self.setWindowTitle("Force-Velocity Profiling")
        self.setFixedSize(800, 600)
        self.setWindowIcon(QtGui.QIcon('resources/logo.png'))

        self.main_layout = QtWidgets.QVBoxLayout()

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.main_layout.addWidget(self.canvas)

        self.status_label = QtWidgets.QLabel("Статус: Ожидание...")
        self.main_layout.addWidget(self.status_label)

        self.back_button = QtWidgets.QPushButton("Вернуться на главную")
        self.back_button.clicked.connect(self.return_to_main)
        self.main_layout.addWidget(self.back_button)

        central_widget = QtWidgets.QWidget(self)
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

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

    def on_new_data(self, forces, velocities):
        self.forces = forces
        self.velocities = velocities

    def update_plot_periodically(self):
        if self.forces and self.velocities:
            self.canvas.update_plot(self.forces, self.velocities)

    def update_status(self, status):
        self.status_label.setText(f"Статус: {status}")

    def return_to_main(self):
        self.tracker.save_to_csv()
        self.worker.stop()
        self.close()
        self.return_to_main_signal.emit()

    def closeEvent(self, event):
        self.worker.stop()
        super().closeEvent(event)
