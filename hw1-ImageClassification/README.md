# HW1 Image Classification

Reproducing Submission
--
1. Install requirements

    | Name        | Version |
    | ----------- | ------- |
    | Python      | 3.7.4   |
    | pytorch     | 1.3.1   |
    | torchvision | 0.4.2   |
    | pillow      | 6.2.0   |

2. Clone the [repo](https://github.com/samuelyutt/Selected-Topics-in-Visual-Recognition-using-Deep-Learning-course.git) from GitHub
    ```
    git clone https://github.com/samuelyutt/Selected-Topics-in-Visual-Recognition-using-Deep-Learning-course.git
    ```

3. Download the [pretrained models](https://drive.google.com/drive/folders/1IHn3exOXiBThCMD13yvZ2LIpGqV8nQUN?usp=sharing) and move them into `checkpoints/RELEASE/`

4. The directory hierarchy should look like this
    ```
    hw1-ImageClassification/
    ├── 2021VRDL_HW1_datasets/
    │   ├── testing_images/
    │   │   ├── 0001.jpg
    │   │   └── ...
    │   ├── training_images/
    │   │   ├── 0003.jpg
    │   │   └── ...
    │   ├── classes.txt
    │   ├── testing_img_order.txt
    │   └── training_labels.txt
    ├── checkpoints/
    │   └── RELEASE/
    │       ├── resnet152_130e.pth
    │       ├── resnext101_32x8d_70e.pth
    │       ├── resnext101_32x8d_400e.pth
    │       ├── resnext101_32x8d_advtrsf_160e.pth
    │       └── resnext101_32x8d_advtrsf_nonoised_250e.pth
    └── src/
        ├── inference.py
        └── hw1.ipynb
    ```
6. Execute `inference.py` in `src/`
    ```
    python inference.py
    ```

7. You will find the submission file `answer.txt` in `src/` in 20 minutes if GPU is available. 

