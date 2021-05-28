"""Set up an NN architecture, run its training and test on the diode clipper data."""
from functools import partial
from pathlib import Path
import torch
import torchaudio
from torchdiffeq import odeint, odeint_adjoint
import CoreAudioML.networks as networks
import CoreAudioML.training as training
from NetworkTraining import NetworkTraining, get_run_name, create_dataset
from models import StateTrajectoryNetworkFF, ODENet, ODENetDerivative


if __name__ == '__main__':
    session = NetworkTraining()
    session.dataset = create_dataset(validation_frame_len=22050, test_frame_len=22050)
    sampling_rate = session.dataset.subsets['test'].fs

    session.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # session.network = networks.SimpleRNN(unit_type="LSTM", hidden_size=8, skip=0)
    # session.network = StateTrajectoryNetworkFF()
    session.network = ODENet(ODENetDerivative(), partial(odeint, method='euler'), dt=1/sampling_rate)
    session.transfer_to_device()
    session.optimizer = torch.optim.Adam(session.network.parameters(), lr=0.001)
    
    model_directory = Path('diode_clipper', 'runs', 'odenet')
    # model_directory = Path('diode_clipper', 'runs', 'stn')
    # session.run_directory = model_directory / 'May27_16-43-39_axel'
    # session.load_checkpoint()
    run_name = get_run_name()
    session.run_directory =  model_directory / run_name

    session.loss = training.ESRLoss()
    
    session.epochs = 20
    session.segments_in_a_batch = 256
    session.samples_between_updates = 2048
    session.initialization_length = 0

    session.run()

    session.device = 'cpu'
    test_output, test_loss = session.test()
    print(f'Test loss: {test_loss}')
    session.writer.add_scalar('Loss/test', test_loss, session.epochs)

    test_output_path = (session.run_directory / 'test_output.wav').resolve()
    torchaudio.save(test_output_path, test_output[None, :, 0, 0], sampling_rate)

    if torch.cuda.is_available():
        session.writer.add_scalar('Maximum GPU memory usage [MB]', torch.cuda.max_memory_allocated('cuda') / (2 ** 20), session.epochs)

    session.writer.close()
