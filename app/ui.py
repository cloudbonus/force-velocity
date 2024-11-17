from PyQt6 import QtWidgets
from plot_window import PlotWindow


class InputWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Введите параметры")
        self.setFixedSize(800, 600)  # Одинаковый размер окна

        self.layout = QtWidgets.QVBoxLayout()

        self.mass_label = QtWidgets.QLabel("Масса (кг):")
        self.mass_input = QtWidgets.QLineEdit()
        self.start_button = QtWidgets.QPushButton("Начать анализ")
        self.start_button.clicked.connect(self.start_analysis)

        self.layout.addWidget(self.mass_label)
        self.layout.addWidget(self.mass_input)
        self.layout.addWidget(self.start_button)

        self.setLayout(self.layout)

    def start_analysis(self):
        try:
            mass = float(self.mass_input.text())
            if mass <= 0:
                raise ValueError("Масса должна быть положительным числом.")
            video_path = "../dataset/pose_movement/jump.mp4"  # Путь к видео
            model_path = "../model/pose_movement/heavy.task"  # Путь к модели
            self.plot_window = PlotWindow(mass, video_path, model_path)
            self.plot_window.show()
            self.close()
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Ошибка", "Введите корректное значение массы.")

