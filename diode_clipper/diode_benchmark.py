"""Set up an LSTM-RNN architecture, run its training and test on the diode clipper data."""
import socket
from datetime import datetime
from pathlib import Path
import torch
import torchaudio
from torch.utils.tensorboard import SummaryWriter
import CoreAudioML.networks as networks
import CoreAudioML.training as training
import CoreAudioML.dataset as dataset
from NetworkTraining import NetworkTraining
from models.StateTrajectoryNetwork import StateTrajectoryNetworkFF


def get_run_name():
    return datetime.now().strftime(r"%B%d_%H-%M-%S") + f'_{socket.gethostname()}'

def create_dataset():
    d = dataset.DataSet(data_dir=str(Path('diode_clipper', 'data').resolve()))

    d.create_subset('train', frame_len=22050)
    d.create_subset('validation')
    d.create_subset('ignore')
    d.load_file('diodeclip', set_names=['train', 'validation', 'ignore'], splits=[0.8*0.8, 0.8*0.2, (1.0 - 0.8*0.8 - 0.8*0.2)])

    d.create_subset('test')
    d.load_file('test', set_names='test')

    return d

if __name__ == '__main__':
    run_name = get_run_name() + '_lr01'
    # run_directory = Path('diode_clipper', 'runs', 'lstm', run_name)
    run_directory = Path('diode_clipper', 'runs', 'stn', run_name)
    
    session = NetworkTraining()
    
    session.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # session.network = networks.SimpleRNN(unit_type="LSTM", hidden_size=8, skip=0)
    session.network = StateTrajectoryNetworkFF()
    session.optimizer = torch.optim.Adam(session.network.parameters(), lr=0.01)
    session.loss = training.ESRLoss()
    
    session.dataset = create_dataset()

    session.epochs = 50
    session.segments_in_a_batch = 256
    session.samples_between_updates = 2048
    session.initialization_length = 1000
    session.model_store_path = (run_directory / 'stn_3x4_tf.pth').resolve()
    session.writer = SummaryWriter(run_directory)

    session.run()

    session.device = 'cpu'
    session.network.load_state_dict(torch.load(session.model_store_path))

    test_output, test_loss = session.test()
    print(f'Test loss: {test_loss}')
    session.writer.add_scalar('Loss/test', test_loss, session.epochs)

    test_output_path = (run_directory / 'test_output.wav').resolve()
    torchaudio.save(test_output_path, test_output[None, :, 0, 0], session.dataset.subsets['test'].fs)

    if torch.cuda.is_available():
        session.writer.add_scalar('Maximum GPU memory usage', torch.cuda.max_memory_allocated('cuda'), session.epochs)

    session.writer.close()
