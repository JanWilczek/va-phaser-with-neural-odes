import torch
import math


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

    def run(self):
        """Run full network training."""
        self.transfer_to_device()

        for epoch in range(self.epochs):
            epoch_loss = self.train_epoch()
            print(f'Epoch: {epoch + 1}/{self.epochs}; Train loss: {epoch_loss}')
            self.writer.add_scalar('Loss/train', epoch_loss, epoch + 1)

        self.writer.flush()

    def train_epoch(self):
        segments_order = torch.randperm(self.segments_count)
        epoch_loss = 0.0

        for i in range(self.minibatch_count):
            minibatch_segment_indices = segments_order[i*self.segments_in_a_batch:(i+1)*self.segments_in_a_batch]
            input_minibatch = self.input_data('train')[:, minibatch_segment_indices, :].to(self.device)
            target_minibatch = self.target_data('train')[:, minibatch_segment_indices, :].to((self.device))

            with torch.no_grad():
                self.network(input_minibatch[0:self.initialization_length, :, :])

            subsegment_start = self.initialization_length
            minibatch_loss = 0.0

            for subsequence_id in range(self.subsegments_count):
                output = self.network(input_minibatch[subsegment_start:subsegment_start + self.samples_between_updates, :, :])

                loss = self.loss(output, target_minibatch[subsegment_start:subsegment_start + self.samples_between_updates, :, :])
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.network.detach_hidden()

                subsegment_start += self.samples_between_updates
                epoch_loss += loss.item()

                torch.save(self.network.state_dict(), self.model_store_path)
                
            self.network.reset_hidden()
        return epoch_loss / (self.segment_length * self.segments_count)

    def test(self):
        with torch.no_grad():
            output = self.network(self.input_data('test'))
            loss = self.loss(output, self.target_data('test')).item()
        return output, loss

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
