from PyQt6 import QtWidgets, QtGui, QtCore


class InputWindow(QtWidgets.QWidget):
    # Define a custom signal
    start_analysis_signal = QtCore.pyqtSignal(float, str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Force-Velocity Profiling")
        self.setFixedSize(800, 600)
        self.setWindowIcon(QtGui.QIcon('resources/logo.png'))

        # Main layout
        self.layout = QtWidgets.QVBoxLayout()

        # Header
        header_label = QtWidgets.QLabel("Force-Velocity Profiling")
        header_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet("font-size: 36px; font-weight: bold;")
        self.layout.addWidget(header_label)

        # Input fields
        self.mass_label = QtWidgets.QLabel("Масса (кг):")
        self.mass_input = QtWidgets.QLineEdit()
        self.mass_input.setFixedWidth(200)

        # Start button
        self.start_button = QtWidgets.QPushButton("Начать анализ")
        self.start_button.setFixedWidth(200)
        self.start_button.clicked.connect(self.start_analysis)

        # Form layout
        form_layout = QtWidgets.QVBoxLayout()
        form_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        form_layout.addWidget(self.mass_label)
        form_layout.addWidget(self.mass_input)
        form_layout.addWidget(self.start_button)

        # Add form to main layout
        self.layout.addLayout(form_layout)
        self.setLayout(self.layout)

    def start_analysis(self):
        try:
            mass = float(self.mass_input.text())
            if mass <= 0:
                raise ValueError("Масса должна быть положительным числом.")

            video_path = "../dataset/pose_movement/jump.mp4"
            model_path = "../model/pose_movement/heavy.task"

            # Emit signal with parameters (mass, video_path, model_path)
            self.start_analysis_signal.emit(mass, video_path, model_path)

        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Ошибка", "Введите корректное значение массы.")
