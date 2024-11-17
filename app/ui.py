from PyQt6 import QtWidgets, QtGui, QtCore

from plot_window import PlotWindow


class InputWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Force-Velocity Profiling")
        self.setFixedSize(800, 600)

        self.setWindowIcon(QtGui.QIcon('resources/logo.png'))
        # Создаем основной layout
        self.layout = QtWidgets.QVBoxLayout()

        # Заголовок с иконкой
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(20, 0, 20, 150)  # Отступы для всей верхней части

        # Заголовок (Force-Velocity Profiling)
        header_label = QtWidgets.QLabel("Force-Velocity Profiling")
        header_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # Выравнивание по центру
        header_label.setStyleSheet("font-size: 36px; font-weight: bold;")  # Увеличенный шрифт и жирный

        # Добавляем заголовок в header_layout
        header_layout.addWidget(header_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Добавляем заголовок в основной layout
        self.layout.addLayout(header_layout)

        # Место для ввода массы
        self.mass_label = QtWidgets.QLabel("Масса (кг):")
        self.mass_input = QtWidgets.QLineEdit()
        self.mass_input.setFixedWidth(200)  # Устанавливаем фиксированную ширину для поля ввода

        # Кнопка для старта анализа
        self.start_button = QtWidgets.QPushButton("Начать анализ")
        self.start_button.setFixedWidth(200)  # Устанавливаем фиксированную ширину для кнопки
        self.start_button.clicked.connect(self.start_analysis)

        # Создаем контейнер для ввода массы и кнопки
        form_layout = QtWidgets.QVBoxLayout()
        form_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # Выравнивание по центру

        # Добавляем метку и поле ввода
        form_layout.addWidget(self.mass_label)
        form_layout.addWidget(self.mass_input)
        form_layout.addWidget(self.start_button)

        # Добавляем form_layout в основной layout
        self.layout.addLayout(form_layout)

        # Выравнивание всех элементов по центру
        self.layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Устанавливаем layout
        self.setLayout(self.layout)

    def start_analysis(self):
        try:
            mass = float(self.mass_input.text())
            if mass <= 0:
                raise ValueError("Масса должна быть положительным числом.")
            video_path = "../dataset/pose_movement/jump.mp4"  # Путь к видео
            model_path = "../model/pose_movement/heavy.task"  # Путь к модели
            self.plot_window = PlotWindow(self, mass, video_path, model_path)  # Передаем родительское окно
            self.plot_window.show()
            self.close()
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Ошибка", "Введите корректное значение массы.")





