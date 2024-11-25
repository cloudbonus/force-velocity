import sys

from PyQt6 import QtWidgets

from ui.input_window import InputWindow
from ui.plot_window import PlotWindow


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.input_window = InputWindow()
        self.input_window.start_analysis_signal.connect(self.show_plot_window)
        self.plot_window = None

    def show_input_window(self):
        self.input_window.show()

    def show_plot_window(self, mass, video_path, model_path):
        self.plot_window = PlotWindow(mass, video_path, model_path)
        self.plot_window.return_to_main_signal.connect(self.show_input_window)
        self.plot_window.show()
        self.input_window.close()


def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show_input_window()  # Start by showing the input window
    sys.exit(app.exec())


if __name__ == "__main__":
    main()