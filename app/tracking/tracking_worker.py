from PyQt6.QtCore import QThread, pyqtSignal

from app.tracking.jump_tracker import JumpData


class TrackingWorker(QThread):
    data_ready = pyqtSignal(JumpData)
    status_update = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
        self.running = True

    def run(self):
        while self.running:
            data = self.tracker.update()
            if data:
                self.data_ready.emit(data)
                self.status_update.emit("Обработка данных...")
            else:
                break

        self.status_update.emit("Обработка завершена")
        self.finished.emit()

    def stop(self):
        self.running = False
        self.wait()
