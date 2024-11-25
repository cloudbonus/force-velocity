import time

from PyQt6.QtCore import QThread, pyqtSignal


class TrackingWorker(QThread):
    data_ready = pyqtSignal(list, list)
    status_update = pyqtSignal(str)

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
                self.status_update.emit("Идет обработка")
            else:
                self.status_update.emit("Обработка завершена")
                time.sleep(1)

    def stop(self):
        self.running = False
        self.wait()
