from PyQt6.QtCore import QThread, pyqtSignal
import time

class TrackingWorker(QThread):
    data_ready = pyqtSignal(list, list)  # Сигнал для передачи данных: forces и velocities
    status_update = pyqtSignal(str)  # Сигнал для обновления статуса

    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
        self.running = True
        self.processing_done = False  # Индикатор завершения обработки

    def run(self):
        while self.running:
            data = self.tracker.update()
            if data:
                forces, velocities = data
                self.data_ready.emit(forces, velocities)
                self.status_update.emit("Идет обработка")  # Обновляем статус на "Идет обработка"
            else:
                if not self.processing_done:  # Если данные закончились, меняем статус
                    self.processing_done = True
                    self.status_update.emit("Обработка закончена")  # Статус завершен
                time.sleep(1)  # Пауза, чтобы не перегружать CPU

    def stop(self):
        self.running = False
        self.wait()




