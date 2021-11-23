# HW2 Object Detection

**Note that this work referred to the following sources:**
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [Street-View-House-Numbers-Detection](https://github.com/chia56028/Street-View-House-Numbers-Detection)

Reproducing Submission
--
1. Download or copy [inference.py](https://colab.research.google.com/drive/1z0spJDJvMnFap-RAFbM8qMNmKQWyJ723?usp=sharing) and open it in [Google Colab](https://colab.research.google.com)
2. Execute every cell in the notebook
3. You will find the submission file `answer.json` in `src/yolov5/`

Reproducing Training
--
1. Clone the [repo](https://github.com/samuelyutt/Selected-Topics-in-Visual-Recognition-using-Deep-Learning-course.git) from GitHub
    ```
    $ git clone https://github.com/samuelyutt/Selected-Topics-in-Visual-Recognition-using-Deep-Learning-course.git
    ```

2. Download [training data](https://drive.google.com/file/d/1DrL7iqM43q7uoV9zDt5av59lAb2286VI/view?usp=sharing) and [testing data](https://drive.google.com/file/d/1_UmIKmOpEmIdROzkfWYXkiEI6C40a5NG/view?usp=sharing)

3. Download configuration files [svhn.yaml](https://drive.google.com/file/d/1RzJKh3A5HhtNopKdDC5M-7n0iYA1lpiJ/view?usp=sharing) and [hyp.finetune.yaml](https://drive.google.com/file/d/1KYMBxMg2Bx0QatqXSK4TJtCT6GMchQyP/view?usp=sharing)


4. Make the directory hierarchy look like this
    ```
    hw2-ObjectDetection/
    ├── datasets/
    │   └── svhn/
    │       ├── hyp.finetune.yaml
    │       ├── svhn.yaml
    │       ├── images/
    │       │   ├── train/
    │       │   ├── val/
    │       │   └── test/
    │       ├── labels/
    │       │   ├── train/
    │       │   └── val/
    │       └── origin/
    │           ├── test.zip
    │           └── train.zip
    └── src/
        ├── prepare.py
        └── yolov5/
            ├── train.py
            ├── val.py
            ├── requirements.txt
            └── ...
    ```


5. Install requirements
    1. Basic requirements

        | Name   | Version |
        | ------ | ------- |
        | Python | 3.8.3   |
        | h5py   | 2.10.0  |
        | numpy  | 1.18.5  |
        | Pillow | 7.2.0   |

    2. YOLOv5 requirements
        ```
        $ cd hw2-ObjectDetection/src/yolov5/
        $ pip install -r requirements.txt
        ```

6. Prepare for training data by executing `prepare.py`
    ```
    $ cd hw2-ObjectDetection/datasets/svhn/origin/
    $ unzip train.zip
    $ cd ../../../src/
    $ python prepare.py
    ```

7. Prepare for testing data by copying data
    ```
    $ cd hw2-ObjectDetection/datasets/svhn/origin/
    $ unzip test.zip
    $ cp -a test/. ../images/test/
    ```

8. Training
    ```
    $ cd hw2-ObjectDetection/src/yolov5/
    $ python train.py --img 640 --batch 64 --epochs 150 --data ../../datasets/svhn/svhn.yaml --weights yolov5x6.pt --hyp ../../datasets/svhn/hyp.finetune.yaml --device 0,1,2,3
    ```
    If CUDA is out of memory, lower the batch size and try again.

9. Testing
    ```
    $ python val.py --data ../../datasets/svhn/svhn.yaml --batch-size 64 --img 640 --task test --device 0,1,2,3 --save-json --weights runs/train/exp/weights/best.pt
    ```
