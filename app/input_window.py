from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QIcon

from PyQt6.QtWidgets import QFileDialog, QMessageBox


class InputWindow(QtWidgets.QWidget):
    start_analysis_signal = QtCore.pyqtSignal(float, str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Force-Velocity Profiling")
        self.setFixedSize(1280, 720)
        self.setWindowIcon(QIcon('resources/logo.png'))

        self.layout = QtWidgets.QVBoxLayout()

        header_label = QtWidgets.QLabel("Force-Velocity Profiling")
        header_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet("font-size: 36px; font-weight: bold;")
        self.layout.addWidget(header_label)

        self.mass_label = QtWidgets.QLabel("Масса (кг):")
        self.mass_input = QtWidgets.QLineEdit()
        self.mass_input.setFixedWidth(200)

        self.video_source_group = QtWidgets.QGroupBox("Выберите источник видео:")
        self.video_source_layout = QtWidgets.QVBoxLayout()

        self.file_radio = QtWidgets.QRadioButton("Использовать файл")
        self.file_radio.setChecked(True)
        self.camera_radio = QtWidgets.QRadioButton("Использовать камеру")

        self.video_source_layout.addWidget(self.file_radio)
        self.video_source_layout.addWidget(self.camera_radio)
        self.video_source_group.setLayout(self.video_source_layout)

        self.file_select_button = QtWidgets.QPushButton("Выбрать видеофайл")
        self.file_select_button.setVisible(True)
        self.file_select_button.clicked.connect(self.select_file)

        self.selected_file_label = QtWidgets.QLabel("Выбранный файл: Не выбран")
        self.selected_file_label.setWordWrap(True)
        self.selected_file_label.setVisible(True)

        self.start_button = QtWidgets.QPushButton("Начать анализ")
        self.start_button.setFixedWidth(200)
        self.start_button.clicked.connect(self.start_analysis)

        form_layout = QtWidgets.QVBoxLayout()
        form_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        form_layout.addWidget(self.mass_label)
        form_layout.addWidget(self.mass_input)
        form_layout.addWidget(self.video_source_group)
        form_layout.addWidget(self.file_select_button)
        form_layout.addWidget(self.selected_file_label)
        form_layout.addWidget(self.start_button)

        self.layout.addLayout(form_layout)
        self.setLayout(self.layout)

        self.file_radio.toggled.connect(self.toggle_file_input)

        self.video_file_path = None

    def toggle_file_input(self):
        if self.file_radio.isChecked():
            self.file_select_button.setVisible(True)
            self.selected_file_label.setVisible(True)
        else:
            self.file_select_button.setVisible(False)
            self.selected_file_label.setVisible(False)
            self.video_file_path = None
            self.selected_file_label.setText("Выбранный файл: Не выбран")

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видеофайл",
            "",
            "Видео файлы (*.mp4 *.avi *.mov);;Все файлы (*)"
        )
        if file_path:
            self.video_file_path = file_path
            self.selected_file_label.setText(f"Выбранный файл: {file_path}")
        else:
            self.selected_file_label.setText("Выбранный файл: Не выбран")

    def start_analysis(self):
        try:
            mass = float(self.mass_input.text())
            if mass <= 0:
                raise ValueError("Масса должна быть положительным числом.")

            if not self.file_radio.isChecked() and self.camera_radio.isChecked():
                video_source = "0"
            else:
                video_source = self.video_file_path

            model_path = "../model/heavy.task"
            self.start_analysis_signal.emit(mass, video_source, model_path)

        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", str(e))
