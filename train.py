import os
import inyaml
from plan import configure_dataset
from utils import console


def parse_args():
    parser = console.ArgumentParser()
    parser.add_argument('--config', type=str)
    args, unknown_args = parser.parse_known_args()
    args.command = unknown_args
    return args


def main(args):
    with open(args.config, 'r') as file:
        config = inyaml.load(file)
    # env = os.environ.copy()
    configure_dataset(os.environ, config)
    
    devices = config.settings.pop('devices', None)
    if devices is not None:
        if isinstance(devices, int):
            device_str = f'{devices}'
        elif isinstance(devices, list):
            device_str = ",".join(map(str, devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = device_str
    
    identity = config.settings.pop('identity', None)
    if identity is None:
        identity = args.command.pop(0)
    elif isinstance(identity, int):
        identity = str(identity)
    configuration = config.settings.pop('configuration', None)
    if configuration is None:
        configuration = args.command.pop(0)
    fold = config.settings.pop('fold', None)
    if fold is None:
        fold = args.command.pop(0)
    elif isinstance(fold, int):
        fold = str(fold)
    
    argv = [identity, configuration, fold, *args.command]
    from nnunetv2.run.run_training import run_training_entry
    console.run_out[run_training_entry, argv]()


if __name__ == '__main__':
    args = parse_args()
    main(args)