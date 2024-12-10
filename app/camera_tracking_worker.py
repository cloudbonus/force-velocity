from PyQt6.QtCore import QThread, pyqtSignal
from time import time, sleep

from jump_tracker import JumpData


class CameraTrackingWorker(QThread):
    data_ready = pyqtSignal(JumpData)
    status_update = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
        self.running = True
        self.last_update_time = time()

    def run(self):
        status = "Обработка завершена"
        while self.running:
            try:
                data = self.tracker.update()
            except:
                data = None

            if data:
                print(data)
                self.last_update_time = time()
                self.data_ready.emit(data)
            else:
                if time() - self.last_update_time > 10:
                    self.running = False
                    status = "Нет данных в течение 10 секунд. Завершение обработки."
                    break

            sleep(0.1)

        self.status_update.emit(status)
        self.finished.emit()

    def stop(self):
        self.running = False
        self.wait()
