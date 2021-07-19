import json
import torch
import torchaudio
import socket
from datetime import datetime


def attach_scheduler(args, session):
    if args.one_cycle_lr is not None:
        session.scheduler = torch.optim.lr_scheduler.OneCycleLR(session.optimizer,
                                                                max_lr=args.one_cycle_lr,
                                                                div_factor=(args.one_cycle_lr / args.learn_rate),
                                                                final_div_factor=20,
                                                                epochs=(session.epochs - session.epoch),
                                                                steps_per_epoch=session.minibatch_count,
                                                                last_epoch=(session.epoch - 1),
                                                                cycle_momentum=False)
    elif args.cyclic_lr is not None:
        session.scheduler = torch.optim.lr_scheduler.CyclicLR(session.optimizer,
                                                              base_lr=args.learn_rate,
                                                              max_lr=args.cyclic_lr,
                                                              step_size_up=250,
                                                              last_epoch=(session.epoch - 1),
                                                              cycle_momentum=False)


def load_checkpoint(args, session, model_directory):
    if args.checkpoint is not None:
        session.run_directory = model_directory / args.checkpoint
        try:
            session.load_checkpoint(best_validation=True)
        except BaseException:
            print("Failed to load a best validation checkpoint. Reverting to the last checkpoint.")
            session.load_checkpoint(best_validation=False)
        if session.scheduler is None:
            # Scheduler's learning rate is not loaded but overwritten
            for param_group in session.optimizer.param_groups:
                param_group['lr'] = args.learn_rate


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def save_args(session, args):
    save_json(vars(args), session.run_directory / 'args.json')
    session.writer.add_text('Command line arguments', json.dumps(vars(args)))


def test(session):
    session.device = 'cpu'
    test_output, test_loss = session.test()
    print(f'Test loss: {test_loss}')
    session.writer.add_scalar('Loss/test', test_loss, session.epochs)

    test_output_path = (session.run_directory / 'test_output.wav').resolve()
    torchaudio.save(test_output_path, test_output[None, :], session.sampling_rate('test'))


def get_teacher_forcing_gate(teacher_forcing_description):
    description_to_gate = {'always': lambda epoch_progress: True,
                           'never': lambda epoch_progress: False,
                           'bernoulli': lambda epoch_progress: torch.bernoulli(torch.Tensor([1 - epoch_progress]))
                           }
    return description_to_gate[teacher_forcing_description]


def log_memory_usage(session):
    if torch.cuda.is_available():
        BYTES_IN_MEGABYTE = 2 ** 20
        session.writer.add_scalar('Maximum GPU memory usage [MB]',
                                  torch.cuda.max_memory_allocated('cuda') / BYTES_IN_MEGABYTE,
                                  session.epochs)

def close_session(session):
    log_memory_usage(session)
    session.writer.close()

def save_json(json_data, filepath):
    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=4)

def get_run_name(suffix=''):
    name = datetime.now().strftime(r"%B%d_%H-%M-%S") + f'_{socket.gethostname()}'
    if len(suffix) > 0:
        name += '_' + suffix
    return name
