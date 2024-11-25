import time

from PyQt6.QtCore import QThread, pyqtSignal


class TrackingWorker(QThread):
    data_ready = pyqtSignal(list, list)
    status_update = pyqtSignal(str)

    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
        self.running = True
        self.video_finished = False  # Новый флаг для отслеживания окончания видео

    def run(self):
        while self.running:
            # Получаем данные из трекера
            data = self.tracker.update()

            # Если данные получены, передаем их на обработку
            if data:
                forces, velocities = data
                self.data_ready.emit(forces, velocities)
                self.status_update.emit("Идет обработка")
            elif not self.video_finished:
                # Если видео не завершилось, продолжаем ожидать
                self.status_update.emit("Ожидание данных (видео завершено)")
                time.sleep(1)
            else:
                # Если видео завершилось и данные тоже обработаны, завершить обработку
                self.status_update.emit("Обработка завершена")
                time.sleep(1)

    def stop(self):
        self.running = False
        self.wait()

    def set_video_finished(self):
        """Этот метод вызывается для пометки видео как завершенного."""
        self.video_finished = True
