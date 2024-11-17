from PyQt6 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from tracking_worker import TrackingWorker
from tracker import ForceVelocityTracker

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class PlotWindow(QtWidgets.QMainWindow):
    def __init__(self, mass, video_path, model_path):
        super().__init__()
        self.setWindowTitle("Force-Velocity Profiling")
        self.setFixedSize(800, 600)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.setCentralWidget(self.canvas)

        self.tracker = ForceVelocityTracker(mass, video_path, model_path)

        self.worker = TrackingWorker(self.tracker)
        self.worker.data_ready.connect(self.receive_data)  # Подключаем сигнал к обработчику
        self.worker.start()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000)  # Обновление каждые 1 сек
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

        self.forces = []
        self.velocities = []

    def receive_data(self, forces, velocities):
        self.forces = forces
        self.velocities = velocities

    def update_plot(self):
        if self.forces and self.velocities:
            self.canvas.axes.cla()
            self.canvas.axes.plot(self.velocities, self.forces, 'o', label="Force-Velocity Data")
            self.canvas.axes.set_xlabel("Velocity (m/s)")
            self.canvas.axes.set_ylabel("Force (N)")
            self.canvas.axes.set_title("Force-Velocity Profile")
            self.canvas.axes.legend()
            self.canvas.axes.grid(True)

            self.canvas.axes.set_ylim(-1000, 1000)
            self.canvas.axes.set_xlim(-2, 2)

            self.canvas.draw()

    def closeEvent(self, event):
        self.worker.stop()
        super().closeEvent(event)


