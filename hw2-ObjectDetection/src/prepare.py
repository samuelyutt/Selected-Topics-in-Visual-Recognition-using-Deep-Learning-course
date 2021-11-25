# Referenced from https://github.com/chia56028/
# Street-View-House-Numbers-Detection/blob/main/data/svhn/mat_to_yolo.py
import h5py
import shutil
import numpy as np
from PIL import Image

DATA_DIR = '../datasets/svhn'
VAL_RATIO = 0.05


def get_name(index, hdf5_data):
    name_ref = hdf5_data['/digitStruct/name'][index].item()
    return ''.join([chr(v[0]) for v in hdf5_data[name_ref]])


def get_bbox(index, hdf5_data):
    attrs = {}
    item_ref = hdf5_data['/digitStruct/bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item_ref][key]
        values = [hdf5_data[attr[i].item()][0][0].astype(int)
                  for i in range(len(attr))] if len(attr) > 1 else [attr[0][0]]
        attrs[key] = values
    return attrs


def main():
    hdf5 = h5py.File(f'{DATA_DIR}/origin/train/digitStruct.mat')

    data_cnt = len(hdf5['/digitStruct/name'])
    val_data_cnt = int(data_cnt * VAL_RATIO)

    print(f'Origin train data counts: {data_cnt}')
    print(f'Train data counts: {val_data_cnt}')
    print(f'Val data counts: {data_cnt - val_data_cnt}')

    for i in range(len(hdf5['/digitStruct/name'])):
        task = 'val' if i < val_data_cnt else 'train'
        img_out_dir = f'{DATA_DIR}/images/{task}'
        lbl_out_dir = f'{DATA_DIR}/labels/{task}'

        img_name = get_name(i, hdf5)
        img = np.array(
            Image.open(f'{DATA_DIR}/origin/train/{img_name}').convert('RGB')
        )
        h, w, c = img.shape
        arr = get_bbox(i, hdf5)

        txt_file = open(
            f'{lbl_out_dir}/{img_name.replace(".png", ".txt")}', 'w'
        )
        for idx in range(len(arr['label'])):
            label = arr['label'][idx]
            if label == 10:
                label = 0
            _l = arr['left'][idx]
            _t = arr['top'][idx]
            _w = arr['width'][idx]
            if (_l + _w) > w:
                _w = w - _l - 1
            _h = arr['height'][idx]
            if (_t + _h) > h:
                _h = h - _t - 1
            x_center = (_l + _w / 2) / w
            y_center = (_t + _h / 2) / h
            bbox_width = _w / w
            bbox_height = _h / h

            s = f'{label} {x_center} {y_center} {bbox_width} {bbox_height}'
            if idx != len(arr['label']) - 1:
                s += '\n'
            txt_file.write(s)
        txt_file.close()

        # Move image to the correct directory
        shutil.copy(
            f'{DATA_DIR}/origin/train/{img_name}', f'{img_out_dir}/{img_name}'
        )

    print('Done')


if __name__ == '__main__':
    main()
