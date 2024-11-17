from PyQt6.QtCore import QThread, pyqtSignal


class TrackingWorker(QThread):
    data_ready = pyqtSignal(list, list)  # Сигнал для передачи данных: forces и velocities

    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
        self.running = True

    def run(self):
        while self.running:
            data = self.tracker.update()
            if data:
                forces, velocities = data
                self.data_ready.emit(forces, velocities)

    def stop(self):
        self.running = False
        self.wait()
