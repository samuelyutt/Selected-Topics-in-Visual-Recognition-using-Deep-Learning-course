import os
import shutil
import json
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

DATA_DIR = '../dataset'
OUT_DIR = './Swin-Transformer-Object-Detection/data/nuclei'
VAL_CNT = 1


def main():
    in_data_types = ['train', 'test']
    out_data_types = ['train', 'val', 'test']
    annotataion_dict = {
        out_data_type: {
            'images': [],
            'annotations': [],
            'categories': [{
                'id': 1,
                'name': 'nuclei',
                'supercategory': 'nuclei'
            }]
        } for out_data_type in out_data_types
    }

    for in_data_type in in_data_types:
        IN_DATA_DIR = f'{DATA_DIR}/{in_data_type}'
        image_idx = 0
        mask_idx = 0

        for dir in os.listdir(IN_DATA_DIR):
            if 'TCGA' not in dir:
                continue

            out_data_type = ('test' if in_data_type == 'test' else
                             'val' if image_idx < VAL_CNT else 'train')

            img_name = (f'{dir}' if in_data_type == 'test' else
                        f'{dir}.png')
            img_dir = (f'{IN_DATA_DIR}/{img_name}'
                       if in_data_type == 'test' else
                       f'{IN_DATA_DIR}/{dir}/images/{img_name}')

            img = np.array(Image.open(img_dir))
            shutil.copy(
                img_dir, f'{OUT_DIR}/{out_data_type}/{img_name}'
            )

            annotataion_dict[out_data_type]['images'].append({
                'id': image_idx,
                'width': img.shape[1],
                'height': img.shape[0],
                'file_name': f'{img_name}',
            })

            if in_data_type == 'test':
                continue

            for mask_name in os.listdir(f'{IN_DATA_DIR}/{dir}/masks'):
                if 'mask' not in mask_name:
                    continue

                mask = np.array(
                    Image.open(f'{IN_DATA_DIR}/{dir}/masks/{mask_name}')
                )
                bimask = np.asfortranarray(mask)

                annotation_info = pycococreatortools.create_annotation_info(
                    mask_idx,
                    image_idx,
                    {'id': 1, 'is_crowd': 0},
                    bimask,
                    bimask.shape,
                    tolerance=2
                )

                annotataion_dict[out_data_type]['annotations'].append(
                    annotation_info
                )

                mask_idx += 1
            image_idx += 1

    for out_data_type, values in annotataion_dict.items():
        if out_data_type == 'test':
            continue
        with open(
            f'{OUT_DIR}/{out_data_type}/annotation.json', 'w+'
        ) as json_file:
            if out_data_type == 'test':
                del values['annotations']
            json.dump(values, json_file)


if __name__ == '__main__':
    main()
