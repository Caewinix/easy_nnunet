import os
import inyaml
from utils import console


def configure_dataset(env, config):
    for key, value in config.dataset.items():
        env[f'nnUNet_{key}'] = value


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
    
    from nnunetv2.utilities.dataset_name_id_conversion import convert_dataset_name_to_id
    identity = config.settings.pop('identity', None)
    if identity is None:
        if args.command[0] == '-d':
            identity = args.command[1]
            args.command = args.command[2:]
        else:
            raise RuntimeError("the first argument should be '-d {dataset_id}', or 'identity' in the config should be given")
    else:
        if isinstance(identity, str):
            identity = convert_dataset_name_to_id(identity)
        identity = str(identity)
    argv = ['-d', identity, *args.command]
    
    from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry
    console.run_out[plan_and_preprocess_entry, argv]()


if __name__ == '__main__':
    args = parse_args()
    main(args)