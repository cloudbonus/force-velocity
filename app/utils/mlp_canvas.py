import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.interpolate import interp1d


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def update_plot(self, forces, velocities):
        self.axes.cla()
        self.axes.set_ylim(-1000, 1000)
        self.axes.set_xlim(-2, 2)
        self.axes.plot(velocities, forces, 'o', markersize=2, label="Force-Velocity Data")

        if len(forces) > 2:
            try:
                sorted_indices = np.argsort(velocities)
                sorted_velocities = np.array(velocities)[sorted_indices]
                sorted_forces = np.array(forces)[sorted_indices]

                linear_interp = interp1d(sorted_velocities, sorted_forces, kind='linear', fill_value="extrapolate")
                velocity_range = np.linspace(min(sorted_velocities), max(sorted_velocities), 200)
                force_range = linear_interp(velocity_range)

                self.axes.plot(velocity_range, force_range, 'r-', label="Linear Interpolation")
            except Exception as e:
                print(f"Interpolation error: {e}")

        self.axes.set_xlabel("Velocity (m/s)")
        self.axes.set_ylabel("Force (N)")
        self.axes.set_title("Force-Velocity Profile")
        self.axes.legend()
        self.axes.grid(True)
        self.draw()
