import time
from torch.utils.tensorboard import SummaryWriter


class TrainingTimeLogger:
    def __init__(self, writer: SummaryWriter, epoch=0):
        self.writer = writer
        self.total_time = 0
        self.epoch_count = epoch
        self.epoch_start_time = None

    def epoch_started(self):
        self.epoch_start_time = time.time()

    def epoch_ended(self):
        self.epoch_count += 1
        epoch_duration = time.time() - self.epoch_start_time
        self.total_time += epoch_duration
        average_epoch_duration = self.total_time / self.epoch_count

        self.writer.add_scalar('Total training time [s]', self.total_time, self.epoch_count)
        self.writer.add_scalar('Average epoch duration [s]', average_epoch_duration, self.epoch_count)

        self.writer.flush()
