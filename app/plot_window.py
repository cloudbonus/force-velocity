import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from tracker import ForceVelocityTracker
from tracking_worker import TrackingWorker


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class PlotWindow(QtWidgets.QMainWindow):
    def __init__(self, main_window, mass, video_path, model_path):
        super().__init__()
        self.setWindowIcon(QtGui.QIcon('resources/logo.png'))
        self.setWindowTitle("Force-Velocity Profiling")
        self.setFixedSize(800, 600)

        # Create a vertical layout to hold both the canvas and the button
        self.main_layout = QtWidgets.QVBoxLayout()

        # Create the canvas and add it to the layout
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.main_layout.addWidget(self.canvas)

        # Create the "Back to Main" button
        self.back_button = QtWidgets.QPushButton("Вернуться на главную")
        self.back_button.clicked.connect(self.return_to_main)
        self.main_layout.addWidget(self.back_button)

        # Create a central widget and set the layout
        central_widget = QtWidgets.QWidget(self)
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        # Tracker setup
        self.tracker = ForceVelocityTracker(mass, video_path, model_path)
        self.worker = TrackingWorker(self.tracker)
        self.worker.data_ready.connect(self.receive_data)
        self.worker.start()

        # Timer for updating the plot
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000)  # Update every 1 second
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

        self.forces = []
        self.velocities = []

        # Store the reference to the main window
        self.main_window = main_window

    def receive_data(self, forces, velocities):
        self.forces = forces
        self.velocities = velocities

    def update_plot(self):
        if self.forces and self.velocities:
            # Clear the axes before plotting new data
            self.canvas.axes.cla()

            # Plot the original force-velocity data
            self.canvas.axes.plot(self.velocities, self.forces, 'o', label="Force-Velocity Data")

            # Set the labels and title
            self.canvas.axes.set_xlabel("Velocity (m/s)")
            self.canvas.axes.set_ylabel("Force (N)")
            self.canvas.axes.set_title("Force-Velocity Profile")
            self.canvas.axes.legend()
            self.canvas.axes.grid(True)

            # Set the limits for the axes
            self.canvas.axes.set_ylim(-1000, 1000)
            self.canvas.axes.set_xlim(-2, 2)

            # Check if there are enough data points for polynomial fitting
            if len(self.forces) > 2:
                try:
                    # Fit a polynomial (degree 2) to the filtered force-velocity data
                    poly_coeffs = np.polyfit(self.forces, self.velocities, deg=2)
                    poly_func = np.poly1d(poly_coeffs)

                    # Generate smoothed values for the fitted polynomial curve
                    force_range = np.linspace(min(self.forces), max(self.forces), 200)
                    velocity_range = poly_func(force_range)

                    # Plot the new fitted polynomial curve
                    self.canvas.axes.plot(velocity_range, force_range, 'r-', label="Polynomial Fit")
                except ValueError:
                    # Ignore errors if the fitting is not possible (e.g., not enough variation in data)
                    pass

            # Draw the updated plot
            self.canvas.draw()

    def return_to_main(self):
        self.tracker.save_to_csv()
        self.close()  # Close the current window
        self.main_window.show()  # Show the main window again

    def closeEvent(self, event):
        self.worker.stop()
        super().closeEvent(event)
