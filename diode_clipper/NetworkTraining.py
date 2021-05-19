import torch
import torchaudio
import math
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class NetworkTraining:
    def __init__(self):
        self.epoch = 0
        self.device = 'cpu'
        self.network = None
        self.optimizer = None
        self.dataset = None
        self.loss = None
        self.epochs = -1
        self.segments_in_a_batch = -1
        self.samples_between_updates = -1
        self.initialization_length = -1
        self.enable_teacher_forcing = False
        self.writer = None
        self.__run_directory = None

    def run(self):
        """Run full network training."""
        self.transfer_to_device()
        best_validation_loss = float('inf')

        for self.epoch in range(self.epoch + 1, self.epochs + 1):
            epoch_loss = self.train_epoch()
            validation_output, validation_loss = self.test('validation')

            print(f'Epoch: {self.epoch}/{self.epochs}; Train loss: {epoch_loss}; Validation loss: {validation_loss}.')
            
            self.log_epoch_validation_loss(epoch_loss=epoch_loss, validation_loss=validation_loss)

            torchaudio.save(self.last_validation_output_path, validation_output[None, :, 0, 0].to('cpu'), self.dataset.subsets['validation'].fs)
            
            if validation_loss < best_validation_loss:
                torch.save(self.network.state_dict(), self.best_validation_model_path)
                best_validation_loss = validation_loss


    def train_epoch(self):
        segments_order = torch.randperm(self.segments_count)
        epoch_loss = 0.0

        true_state = self.true_train_state

        for i in range(self.minibatch_count):
            input_minibatch, target_minibatch, true_state_minibatch = self.get_minibatch(i, segments_order, true_state)
            
            should_include_teacher_forcing = self.enable_teacher_forcing and torch.bernoulli(torch.Tensor([1 - i / self.minibatch_count]))

            self.network.reset_hidden()
            with torch.no_grad():
                self.network(input_minibatch[0:self.initialization_length, :, :])

            subsegment_start = self.initialization_length

            for subsequence_id in range(self.subsegments_count):
                
                if should_include_teacher_forcing:
                    self.network.true_state = true_state_minibatch[subsegment_start:subsegment_start + self.samples_between_updates, :, :]

                output = self.network(input_minibatch[subsegment_start:subsegment_start + self.samples_between_updates, :, :])

                loss = self.loss(output, target_minibatch[subsegment_start:subsegment_start + self.samples_between_updates, :, :])
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.network.detach_hidden()

                subsegment_start += self.samples_between_updates
                epoch_loss += loss.item()
            
                self.save_checkpoint()

        return epoch_loss / (self.segment_length * self.segments_count)

    def test(self, subset_name='test'):
        self.transfer_to_device()
        self.network.reset_hidden()
        
        with torch.no_grad():
            output = self.network(self.input_data(subset_name).to(self.device))
            loss = self.loss(output, self.target_data(subset_name).to(self.device)).item()
        return output, loss

    def save_checkpoint(self):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, self.checkpoint_path)

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']

    def log_epoch_validation_loss(self, epoch_loss, validation_loss):
        if self.writer is not None:
            self.writer.add_scalar('Loss/train', epoch_loss, self.epoch)
            self.writer.add_scalar('Loss/validation', validation_loss, self.epoch)
            self.writer.flush()

    def get_minibatch(self, minibatch_index, segments_order, true_state):
        minibatch_segment_indices = segments_order[minibatch_index*self.segments_in_a_batch:(minibatch_index+1)*self.segments_in_a_batch]

        def extract_minibatch_segment_and_transfer_to_device(data):
            return data[:, minibatch_segment_indices, :].to(self.device)

        input_minibatch = extract_minibatch_segment_and_transfer_to_device(self.input_data('train'))
        target_minibatch = extract_minibatch_segment_and_transfer_to_device(self.target_data('train'))
        true_state_minibatch = extract_minibatch_segment_and_transfer_to_device(true_state)

        return input_minibatch, target_minibatch, true_state_minibatch

    @property
    def run_directory(self):
        return self.__run_directory

    @run_directory.setter
    def run_directory(self, directory):
        self.__run_directory = Path(directory)
        self.writer = SummaryWriter(self.__run_directory)

    def transfer_to_device(self):
        self.network.to(self.device)

    def input_data(self, subset_name):
        return self.dataset.subsets[subset_name].data['input'][0]

    def target_data(self, subset_name):
        return self.dataset.subsets[subset_name].data['target'][0]

    @property
    def true_train_state(self):
        # True state at time n is the desired output at time n-1. We therefore
        # delay the training output by 1 sample and insert 0 as the state at time step n=0.
        true_state = torch.roll(self.target_data('train'), shifts=1, dims=0)
        true_state[0, :, :] = 0.0
        return true_state

    @property
    def segments_count(self):
        return self.input_data('train').shape[1]

    @property
    def segment_length(self):
        return self.input_data('train').shape[0]

    @property
    def subsegments_count(self):    
        return int(math.ceil((self.segment_length - self.initialization_length) / self.samples_between_updates))

    @property
    def minibatch_count(self):
        return int(math.ceil(self.segments_count / self.segments_in_a_batch))

    @property
    def checkpoint_path(self):
        return self.run_directory / 'checkpoint.pth'

    @property
    def last_validation_output_path(self):
        return self.run_directory / 'last_validation_output.wav'

    @property
    def best_validation_model_path(self):
        return self.run_directory / 'best_validation_loss_model.pth'
