import os
import cv2
from pydicom import dcmread


def data_preprocess():
    datasets = [(
        './data/rsna-pneumonia-detection-challenge/stage_2_train_images',
        './clahe_data/train'), (
        './data/rsna-pneumonia-detection-challenge/stage_2_test_images',
        './clahe_data/test')
    ]

    for ori_img_path, clahe_img_path in datasets:
        for ori_img_name in os.listdir(ori_img_path):
            img_name = ori_img_name.split('.')[0]

            ds = dcmread(f'{ori_img_path}/{img_name}.dcm')
            np_pixel_array = ds.pixel_array
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl1 = clahe.apply(np_pixel_array)

            cv2.imwrite(f'{clahe_img_path}/{img_name}.jpg', cl1)


def create_dir(P):
    if os.path.isdir(P) is True:
        return 0
    else:
        print('Creating {}'.format(P))
        os.makedirs(P)


def main():
    create_dir('clahe_data')
    create_dir('clahe_data/train')
    create_dir('clahe_data/test')
    data_preprocess()


if __name__ == '__main__':
    main()
