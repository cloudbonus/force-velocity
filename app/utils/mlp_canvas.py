from typing import List, Dict

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter1d

from app.tracking.jump_tracker import JumpState, JumpData


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def update_plot(self, data: List[Dict[JumpState, List[JumpData]]], smooth_sigma=2):
        self.axes.cla()

        takeoff_data = {}
        landing_data = {}

        for segment in data:
            for jump in segment.get(JumpState.TAKEOFF, []):
                takeoff_data.setdefault(jump.velocity, []).append(jump.force)
            for jump in segment.get(JumpState.LANDING, []):
                landing_data.setdefault(jump.velocity, []).append(jump.force)

        def aggregate_data(jump_data):
            x, y = [], []
            for velocity, forces in sorted(jump_data.items()):
                x.append(velocity)
                y.append(np.mean(forces))  # Use np.median(forces) if needed
            return np.array(x), np.array(y)

        takeoff_x, takeoff_y = aggregate_data(takeoff_data)
        landing_x, landing_y = aggregate_data(landing_data)

        takeoff_y_smooth = gaussian_filter1d(takeoff_y, sigma=smooth_sigma)
        landing_y_smooth = gaussian_filter1d(landing_y, sigma=smooth_sigma)

        if len(takeoff_x) > 1:
            self.axes.plot(takeoff_x, takeoff_y_smooth, label="Takeoff", color="green")
            self.axes.fill_between(
                takeoff_x, takeoff_y_smooth, color="green", alpha=0.2, label="Eccentric Phase"
            )
        if len(landing_x) > 1:
            self.axes.plot(landing_x, landing_y_smooth, label="Landing", color="red")
            self.axes.fill_between(
                landing_x, landing_y_smooth, color="red", alpha=0.2, label="Concentric Phase"
            )

        self.axes.axvline(x=0, color='black', linewidth=2, linestyle='--', label="Transition Point (Zero Velocity)")
        self.axes.set_xlabel("Velocity (m/s)")
        self.axes.set_ylabel("Force (N)")
        self.axes.set_title("Smoothed Force-Velocity Profile")
        self.axes.legend()
        self.axes.grid(True)

        self.draw()
