import torch
import torchaudio
import math
from pathlib import Path


class NetworkTraining:
    def __init__(self):
        self.network = None
        self.optimizer = None
        self.dataset = None
        self.loss = None
        self.epochs = -1
        self.segments_in_a_batch = -1
        self.samples_between_updates = -1
        self.initialization_length = -1
        self.model_store_path = None
        self.writer = None
        self.enable_teacher_forcing = False
        self.__run_directory = None

    def run(self):
        """Run full network training."""
        self.transfer_to_device()
        best_validation_loss = float('inf')

        for self.epoch in range(1, self.epochs + 1):
            epoch_loss = self.train_epoch()
            validation_output, validation_loss = self.test('validation')

            print(f'Epoch: {self.epoch}/{self.epochs}; Train loss: {epoch_loss}; Validation loss: {validation_loss}.')
            
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', epoch_loss, self.epoch)
                self.writer.add_scalar('Loss/validation', validation_loss, self.epoch)

                self.writer.flush()

            validation_output_path = (self.run_directory / 'last_validation_output.wav').resolve()
            torchaudio.save(validation_output_path, validation_output[None, :, 0, 0].to('cpu'), self.dataset.subsets['validation'].fs)
            
            if validation_loss < best_validation_loss:
                torch.save(self.network.state_dict(), self.run_directory / 'best_validation_loss_model.pth')
                best_validation_loss = validation_loss

    def train_epoch(self):
        segments_order = torch.randperm(self.segments_count)
        epoch_loss = 0.0

        # True state at time n is the desired output at time n-1. We therefore
        # delay the training output by 1 sample and insert 0 as the state at time step n=0.
        true_state = torch.roll(self.target_data('train'), shifts=1, dims=0)
        true_state[0, :, :] = 0.0

        for i in range(self.minibatch_count):
            minibatch_segment_indices = segments_order[i*self.segments_in_a_batch:(i+1)*self.segments_in_a_batch]

            def extract_minibatch_segment_and_transfer_to_device(data):
                return data[:, minibatch_segment_indices, :].to(self.device)

            input_minibatch = self.input_data('train')[:, minibatch_segment_indices, :].to(self.device)
            target_minibatch = self.target_data('train')[:, minibatch_segment_indices, :].to(self.device)
            true_state_minibatch = true_state[:, minibatch_segment_indices, :].to(self.device)
            
            should_include_teacher_forcing = self.enable_teacher_forcing and torch.bernoulli(torch.Tensor([1 - i / self.minibatch_count]))

            self.network.reset_hidden()
            with torch.no_grad():
                self.network(input_minibatch[0:self.initialization_length, :, :])

            subsegment_start = self.initialization_length
            minibatch_loss = 0.0

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

                torch.save(self.network.state_dict(), self.model_store_path)

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
        }, self.model_store_path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']

    @property
    def run_directory(self):
        return self.__run_directory

    @run_directory.setter
    def run_directory(self, directory):
        self.__run_directory = Path(directory)
        self.__run_directory.mkdir(parents=True, exist_ok=True)

    def transfer_to_device(self):
        self.network.to(self.device)

    def input_data(self, subset_name):
        return self.dataset.subsets[subset_name].data['input'][0]

    def target_data(self, subset_name):
        return self.dataset.subsets[subset_name].data['target'][0]

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
