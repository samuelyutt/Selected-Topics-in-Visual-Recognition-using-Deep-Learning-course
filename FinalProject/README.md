# Final Project

Reproducing Submission
--
1. Create a `conda` environment named `rsna`, activate `rsna`, and install requirements

    | Name        | Version |
    | ----------- | ------- |
    | Python      | 3.8.3   |
    | pytorch     | 1.7.0   |
    | torchvision | 0.8.1   |
    | numpy       | 1.18.5  |
    | pandas      | 1.3.5   |
    | tqdm        | 4.47.0  |
    | pydicom     | 2.2.2   |

2. Clone the [repo](https://github.com/samuelyutt/Selected-Topics-in-Visual-Recognition-using-Deep-Learning-course.git) from GitHub
    ```
    $ git clone https://github.com/samuelyutt/Selected-Topics-in-Visual-Recognition-using-Deep-Learning-course.git
    $ cd FinalProject
    ```

3. Download the [pretrained models](https://drive.google.com/file/d/1N5D4QcQyC7eZ2g3E4ZyBYTa32kfE2sC-/view?usp=sharing) and move them into `checkpoints/RELEASE/`

4. Download the [data](https://drive.google.com/file/d/1IxPauHtrOVJhtybtmFKwQhpNYoBVFnhv/view?usp=sharing) and move them into `data/`

5. The directory hierarchy should look like this
    ```
    FinalProject/
    ├── data/
    │   ├── rsna-pneumonia-detection-challenge/
    │   │   ├── stage_2_test_images/
    │   │   ├── stage_2_train_images/
    │   │   ├── stage_2_train_labels.csv
    │   │   └── ...
    ├── checkpoints/
    │   └── RELEASE/
    │       └── fasterrcnn_resnet50_fpn_pneumonia_detection_size500_dropped_clahe_e50.pth
    ├── inference.ipynb
    ├── FinalProject.ipynb
    └── data_preprocess.py
    ```
6. Execute `data_preprocess.py`
    ```
    $ python data_preprocess.py
    ```

7. Execute every cell in `inference.ipynb` under `rsna` kernel

8. You will find the submission file `answer.txt` in `FinalProject/`
