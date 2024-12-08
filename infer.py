import os
import inyaml
import json
from time import time
from datetime import datetime
import numpy
from plan import configure_dataset
from utils import console


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def print_things(*args, also_print_to_console=True, add_timestamp=True):
    timestamp = time()
    dt_object = datetime.fromtimestamp(timestamp)
    if add_timestamp:
        args = (f"{dt_object}:", *args)
    if also_print_to_console:
        print(*args)


def parse_args():
    parser = console.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('-d', type=str, required=False, default=None)
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans')
    parser.add_argument('-f', type=str, required=False, default='5')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer', help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer')
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
    
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    identity = config.settings.pop('identity', None)
    if identity is None and args.d is not None:
        identity = args.d
    fold = config.settings.pop('fold', None)
    if fold is None:
        fold = args.f
    elif isinstance(fold, int):
        fold = str(fold)
    if isinstance(identity, int):
        identity_name = maybe_convert_to_dataset_name(identity)
    else:
        identity_name = identity
    
    configuration = config.settings.pop('configuration', None)
    if configuration is None:
        configuration = args.command.pop(0)
    
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    dataset_json = read_json(os.path.join(config.dataset.preprocessed, identity_name, 'dataset.json'))
    plans_manager = PlansManager(read_json(os.path.join(config.dataset.preprocessed, identity_name, args.p + '.json')))
    label_manager = plans_manager.get_label_manager(dataset_json)
    
    input_folder = os.path.join(config.dataset.raw, identity_name, 'imagesTs')
    output_folder = os.path.join(config.dataset.results, plans_manager.dataset_name, args.tr + '__' + plans_manager.plans_name + "__" + configuration, 'inferenceTs')
    ground_truth_folder = os.path.join(config.dataset.raw, identity_name, 'labelsTs')

    if not os.path.exists(output_folder) or len(os.listdir(output_folder)) <= len(os.listdir(input_folder)):
        argv = ['-i', input_folder, '-o', output_folder, '-d', identity_name, '-c', configuration, '-p', args.p, '-f', fold, *args.command]
        from nnunetv2.inference.predict_from_raw_data import predict_entry_point
        console.run[predict_entry_point, argv]()
    
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
    metrics = compute_metrics_on_folder(
        ground_truth_folder,
        output_folder,
        os.path.join(output_folder, 'summary.json'),
        plans_manager.image_reader_writer_class(),
        dataset_json["file_ending"],
        label_manager.foreground_regions if label_manager.has_regions else label_manager.foreground_labels,
        label_manager.ignore_label,
        chill=True
    )
    
    print_things("Test complete", also_print_to_console=True)
    print_things("Mean Test Dice: ", (metrics['foreground_mean']["Dice"]), also_print_to_console=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)