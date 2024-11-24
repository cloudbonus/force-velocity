import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.interpolate import interp1d

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

        # Создаем основной layout
        self.main_layout = QtWidgets.QVBoxLayout()

        # Canvas для графика
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.main_layout.addWidget(self.canvas)

        # Статус метка
        self.status_label = QtWidgets.QLabel("Статус: Ожидание...")
        self.main_layout.addWidget(self.status_label)

        # Кнопка для возврата в главное окно
        self.back_button = QtWidgets.QPushButton("Вернуться на главную")
        self.back_button.clicked.connect(self.return_to_main)
        self.main_layout.addWidget(self.back_button)

        # Создаем центральный виджет и устанавливаем layout
        central_widget = QtWidgets.QWidget(self)
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        # Настройка трекера и рабочего потока
        self.tracker = ForceVelocityTracker(mass, video_path, model_path)
        self.worker = TrackingWorker(self.tracker)
        self.worker.data_ready.connect(self.receive_data)
        self.worker.status_update.connect(self.update_status)  # Подключаем сигнал для статуса
        self.worker.start()

        # Таймер для обновления графика
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

        self.forces = []
        self.velocities = []

        # Сохраняем ссылку на главное окно
        self.main_window = main_window

    def receive_data(self, forces, velocities):
        self.forces = forces
        self.velocities = velocities

    def update_status(self, status):
        # Обновляем статус на UI
        self.status_label.setText(f"Статус: {status}")

    def update_plot(self):
        if self.forces and self.velocities:
            # Очищаем график перед обновлением
            self.canvas.axes.cla()

            # Отображаем оригинальные данные с меньшими точками
            self.canvas.axes.plot(self.velocities, self.forces, 'o', markersize=2, label="Force-Velocity Data")

            # Настройки графика
            self.canvas.axes.set_xlabel("Velocity (m/s)")
            self.canvas.axes.set_ylabel("Force (N)")
            self.canvas.axes.set_title("Force-Velocity Profile")
            self.canvas.axes.legend()
            self.canvas.axes.grid(True)

            # Ограничения для осей
            self.canvas.axes.set_ylim(-1000, 1000)
            self.canvas.axes.set_xlim(-2, 2)

            if len(self.forces) > 2:
                try:
                    # Сортировка данных для корректной интерполяции
                    sorted_indices = np.argsort(self.velocities)
                    sorted_velocities = np.array(self.velocities)[sorted_indices]
                    sorted_forces = np.array(self.forces)[sorted_indices]

                    # Линейная интерполяция
                    linear_interp = interp1d(sorted_velocities, sorted_forces, kind='linear', fill_value="extrapolate")
                    velocity_range = np.linspace(min(sorted_velocities), max(sorted_velocities), 200)
                    force_range = linear_interp(velocity_range)

                    # Добавление линии интерполяции
                    self.canvas.axes.plot(velocity_range, force_range, 'r-', label="Linear Interpolation")

                except Exception as e:
                    print(f"Error in interpolation: {e}")

            # Рисуем обновленный график
            self.canvas.draw()

    def return_to_main(self):
        self.tracker.save_to_csv()
        self.close()
        self.main_window.show()

    def closeEvent(self, event):
        self.worker.stop()
        super().closeEvent(event)

# # Sort the data for proper interpolation
# sorted_indices = np.argsort(self.velocities)
# sorted_velocities = np.array(self.velocities)[sorted_indices]
# sorted_forces = np.array(self.forces)[sorted_indices]
#
# # Ensure there are at least two unique velocities
# unique_velocities = np.unique(sorted_velocities)
#
# if len(unique_velocities) > 1:
#     try:
#         # Perform linear interpolation if there are enough unique velocities
#         linear_interp = interp1d(unique_velocities, sorted_forces[:len(unique_velocities)], kind='linear', fill_value="extrapolate")
#
#         # Generate velocity range for the smoothed spline
#         velocity_range = np.linspace(min(unique_velocities), max(unique_velocities), 200)
#         force_range = linear_interp(velocity_range)
#
#         # Plot the linear interpolation curve
#         self.canvas.axes.plot(velocity_range, force_range, 'r-', label="Linear Interpolation")
#     except Exception as e:
#         print(f"Error in interpolation: {e}")
# else:
#     print("Insufficient unique velocities for interpolation.")
