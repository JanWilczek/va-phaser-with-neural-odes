import torch


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

    def run(self):
       for epoch in range(self.epochs):
            self.train_epoch()

    def train_epoch(self):
        segments_order = torch.randperm(self.segments_count)

        for i in range(self.minibatch_count):
            minibatch_segment_indices = segments_order[i*segments_in_a_batch:(i+1)*self.segments_in_a_batch]
            input_minibatch = self.input_data[:, minibatch_segment_indices, :]
            target_minibatch = self.target_data[:, minibatch_segment_indices, :]

            with torch.no_grad():
                self.network(input_minibatch[0:self.initialization_length, :, :])

            

    @property
    def input_data(self):
        return self.dataset.subsets['train'].data['input'][0]

    @property
    def target_data(self):
        return self.dataset.subsets['train'].data['target'][0]

    @property
    def segments_count(self):
        return self.input_data.shape[1]

    @property
    def minibatch_count(self):
        return int(torch.ceil(self.segments_count / self.segments_in_a_batch))
