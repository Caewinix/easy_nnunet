import os
import math
import re
import argparse
import pandas as pd
import random
from tqdm import tqdm
import numpy as np
from utils import libzip


def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('-p', '--path', type=str, default='./methylation.zip')
    parser.add_argument('--images-dirname', type=str, default='images')
    parser.add_argument('--masks-dirname', type=str, default='labels_mass_merge')
    parser.add_argument('--id-format', type=str, default=r"((?P<id>\d+(?:\.\d+)?)-\.?\d+(?:\.\d+)?-\.?\d+(?:\.\d+)?)(?:_(\d+))?")
    parser.add_argument('--output-base-dir', type=str, default='./')
    parser.add_argument('--output-dataset-name', type=str, default='Methylation')
    parser.add_argument('-r', '--ratio', '--train-val-test-ratio', nargs='+', type=int, default=[3, 1])
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-ln', '--link', type=bool, default=True)
    args = parser.parse_args()
    return args


def arrange(filenames, id_format: str):
    dictionary = dict()
    for filename in filenames:
        match = re.search(id_format, filename)
        if match is not None:
            id = match.group('id')
            name = match.group(1)
            if id in dictionary:
                dictionary[id][name] = filename
            else:
                dictionary[id] = {name : filename}
    return dictionary


_suffix_pattern = r'(?P<suffix>\.(?:[a-zA-Z0-9]+)(?:\.(?=[a-zA-Z0-9]*(?=.*[a-zA-Z]))[a-zA-Z0-9]+)*)$'
def filename_suffix(filename):
    match = re.search(_suffix_pattern, filename)
    if match is not None:
        suffix = match.group('suffix')
        if suffix is None:
            suffix = ''
    else:
        suffix = ''
    return suffix


def file_row_id_and_type(row):
    return f"{row['pid']}-{row['stuid'][-4:]}-{row['seid'][-4:]}", row['direction']


_direction_to_english = {
    '全部' : 'all',
    '矢状位' : 'sagittal',
    '轴位' : 'axial'
}


def main(args):
    random.seed(args.seed)
    ratio = np.asarray(args.ratio) / np.sum(args.ratio)
    
    if os.path.exists(args.output_base_dir):
        base_dir_list = os.listdir(args.output_base_dir)
        try:
            first_dataset_id = max(int(re.search(r'Dataset(\d+).*?', dir_path).group(1)) for dir_path in base_dir_list if os.path.isdir(dir_path)) + 1
        except ValueError:
            first_dataset_id = 1
    else:
        first_dataset_id = 1
    
    orginal_images_dir = os.path.join(args.output_base_dir, args.output_dataset_name, 'images')
    os.makedirs(orginal_images_dir, exist_ok=True)
    orginal_masks_dir = os.path.join(args.output_base_dir, args.output_dataset_name, 'masks')
    os.makedirs(orginal_masks_dir, exist_ok=True)
    
    dataset_dir = {
        'all' : os.path.join(args.output_base_dir, f'Dataset{first_dataset_id:03d}_{args.output_dataset_name}'),
        'sagittal' : os.path.join(args.output_base_dir, f'Dataset{first_dataset_id + 1:03d}_{args.output_dataset_name}Sagittal'),
        'axial' : os.path.join(args.output_base_dir, f'Dataset{first_dataset_id + 2:03d}_{args.output_dataset_name}Axial')
    }
    directional_paths = {
        direction : {
            'output_train_val_images_dir' : os.path.join(output_dir, 'imagesTr'),
            'output_train_val_labels_dir' : os.path.join(output_dir, 'labelsTr'),
            'output_test_images_dir' : os.path.join(output_dir, 'imagesTs'),
            'output_test_labels_dir' : os.path.join(output_dir, 'labelsTs')
        } for direction, output_dir in dataset_dir.items()
    }
    for values in directional_paths.values():
        for path in values.values():
            os.makedirs(path, exist_ok=True)
    
    with libzip.ZipFile(args.path, 'r') as zfile:
        with zfile.open('anno_info.csv') as file:
            info = pd.read_csv(file)
            direction_count = info['direction'].value_counts().to_dict()
            file_id_direction = dict(info.apply(file_row_id_and_type, axis=1).values)
        images_dictionary = list(arrange(libzip.iterdir(zfile, args.images_dirname), args.id_format).items())
        masks_dictionary = arrange(libzip.iterdir(zfile, args.masks_dirname), args.id_format)
        train_val_length = int(ratio[0] * len(images_dictionary))
        total_length = sum(len(content) for _, content in images_dictionary)
        cases_length_digit = math.ceil(math.log10(total_length + 1e-6))
        for key, value in direction_count.items():
            direction_count[key] = math.ceil(math.log10(value + 1e-6))
        random.shuffle(images_dictionary)
        train_val_images = images_dictionary[:train_val_length]
        test_images = images_dictionary[train_val_length:]
        total_type_index = sagittal_index = axial_index = 0
        
        with tqdm(total=len(images_dictionary)) as pbar:
            def _save_contents(
                sequence,
                output_images_all_dir,
                output_masks_all_dir,
                output_images_sagittal_dir,
                output_masks_sagittal_dir,
                output_images_axial_dir,
                output_masks_axial_dir,
            ):
                nonlocal total_type_index, sagittal_index, axial_index
                for key, content in sequence:
                    for name, filename in content.items():
                        suffix = filename_suffix(filename)
                        zip_image_file = os.path.join(args.images_dirname, filename)
                        image_path = os.path.join(orginal_images_dir, os.path.basename(filename))
                        libzip.extract(zfile, zip_image_file, image_path)
                        mask_filename = masks_dictionary[key][name]
                        zip_mask_path = os.path.join(args.masks_dirname, mask_filename)
                        mask_path = os.path.join(orginal_masks_dir, os.path.basename(mask_filename))
                        libzip.extract(zfile, zip_mask_path, mask_path)
                        link_image_path = os.path.join(output_images_all_dir, f'ALL_{total_type_index:0{cases_length_digit}d}_{0:04d}{suffix}')
                        link_mask_path = os.path.join(output_masks_all_dir, f'ALL_{total_type_index:0{cases_length_digit}d}{suffix}')
                        abs_image_path = os.path.abspath(image_path)
                        abs_mask_path = os.path.abspath(mask_path)
                        if os.path.lexists(link_image_path):
                            os.remove(link_image_path)
                        os.symlink(abs_image_path, os.path.abspath(link_image_path))
                        if os.path.lexists(link_mask_path):
                            os.remove(link_mask_path)
                        os.symlink(abs_mask_path, os.path.abspath(link_mask_path))
                        total_type_index += 1
                        try:
                            original_direction = file_id_direction[name]
                        except KeyError: ...
                        else:
                            direction = _direction_to_english[original_direction]
                            upper_direction = direction.upper()
                            direction_length = direction_count[original_direction]
                            if direction == 'sagittal':
                                link_image_path = os.path.join(output_images_sagittal_dir, f'{upper_direction}_{sagittal_index:0{direction_length}d}_{0:04d}{suffix}')
                                link_mask_path = os.path.join(output_masks_sagittal_dir, f'{upper_direction}_{sagittal_index:0{direction_length}d}{suffix}')
                                sagittal_index += 1
                            elif direction == 'axial':
                                link_image_path = os.path.join(output_images_axial_dir, f'{upper_direction}_{axial_index:0{direction_length}d}_{0:04d}{suffix}')
                                link_mask_path = os.path.join(output_masks_axial_dir, f'{upper_direction}_{axial_index:0{direction_length}d}{suffix}')
                                axial_index += 1
                            if args.link:
                                if os.path.lexists(link_image_path):
                                    os.remove(link_image_path)
                                os.symlink(abs_image_path, os.path.abspath(link_image_path))
                                if os.path.lexists(link_mask_path):
                                    os.remove(link_mask_path)
                                os.symlink(abs_mask_path, os.path.abspath(link_mask_path))
                            else:
                                libzip.extract(zfile, zip_image_file, link_image_path)
                                libzip.extract(zfile, zip_mask_path, link_mask_path)
                    pbar.update(1)
            
            _save_contents(
                train_val_images,
                directional_paths['all']['output_train_val_images_dir'],
                directional_paths['all']['output_train_val_labels_dir'],
                directional_paths['sagittal']['output_train_val_images_dir'],
                directional_paths['sagittal']['output_train_val_labels_dir'],
                directional_paths['axial']['output_train_val_images_dir'],
                directional_paths['axial']['output_train_val_labels_dir']
            )
            _save_contents(
                test_images,
                directional_paths['all']['output_test_images_dir'],
                directional_paths['all']['output_test_labels_dir'],
                directional_paths['sagittal']['output_test_images_dir'],
                directional_paths['sagittal']['output_test_labels_dir'],
                directional_paths['axial']['output_test_images_dir'],
                directional_paths['axial']['output_test_labels_dir']
            )


if __name__ == '__main__':
    args = parse_args()
    main(args)