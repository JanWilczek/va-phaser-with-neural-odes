import time
from torch.utils.tensorboard import SummaryWriter


class TrainingTimeLogger:
    def __init__(self, writer: SummaryWriter, epoch=0):
        self.writer = writer
        self.start_epoch = epoch
        self.epoch_count = 0
        self.epoch_start_time = None

    def epoch_started(self):
        self.epoch_start_time = time.time()

    def epoch_ended(self):
        self.epoch_count += 1
        epoch_duration = time.time() - self.epoch_start_time
        self.writer.add_scalar('Epoch duration [s]', epoch_duration, self.current_epoch)
        self.writer.flush()

    @property
    def current_epoch(self):
        return self.start_epoch + self.epoch_count
