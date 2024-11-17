import sys
from PyQt6 import QtWidgets
from ui import InputWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    input_window = InputWindow()
    input_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
