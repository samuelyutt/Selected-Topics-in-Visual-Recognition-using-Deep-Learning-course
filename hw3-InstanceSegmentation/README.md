# HW3 Instance Segmentation

**Note that this work referred to the following sources:**
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [Swin Transformer Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)
- [Swin Transformer paper](https://arxiv.org/pdf/2103.14030.pdf)
- [mmdetection](https://github.com/open-mmlab/mmdetection)

Reproducing Submission
--

1. Clone the [repo](https://github.com/samuelyutt/Selected-Topics-in-Visual-Recognition-using-Deep-Learning-course.git) from GitHub
    ```
    $ git clone https://github.com/samuelyutt/Selected-Topics-in-Visual-Recognition-using-Deep-Learning-course.git
    ```

2. Setup the environment
    ```
    $ cd Selected-Topics-in-Visual-Recognition-using-Deep-Learning-course/hw3-InstanceSegmentation/src/Swin-Transformer-Object-Detection

    # Create a conda virtual environment and activate it
    $ conda create -n swintsfm python=3.7 -y
    $ conda activate swintsfm

    # Install PyTorch and torchvision
    $ conda install pytorch torchvision -c pytorch -y

    $ python setup.py install

    # Install MMDetection
    $ pip install openmim
    $ mim install mmcv-full
    
    # Install pycococreatortools
    $ pip install git+git://github.com/waspinator/pycococreator.git@0.2.0
    ```

3. (Optional) Download the [datasets](https://drive.google.com/file/d/1YuH_b7LjiyO9V4zg5CqPh0MtK2Xfb9aW/view?usp=sharing), unzip the file, and put them under `dataset/`

4. Download one of the pretrained weights from [here](https://drive.google.com/drive/folders/1h742JJZHc_mlncJZGjd10DKE3Q3CThZB?usp=sharing), put it into `src/Swin-Transformer-Object-Detection/work_dirs/mask_rcnn_swin_s/`, and name it `RELEASE_best.pth`

5. Make the directory hierarchy look like this
    ```
    hw3-InstanceSegmentation/
    ├── dataset/
    │   ├── test/
    │   │   ├── TCGA-50-5931-01Z-00-DX1.png
    │   │   └── ...
    │   └── train/
    │       ├── TCGA-18-5592-01Z-00-DX1
    │       │   ├── images/
    │       │   │   └── TCGA-18-5592-01Z-00-DX1.png
    │       │   └── masks/
    │       │       ├── mask_0001.png
    │       │       └── ...
    │       └── ...
    └── src/
        ├── inference.ipynb
        ├── prepare.py
        ├── work_dirs/
        │   └── mask_rcnn_swin_s/
        │        └── RELEASE_best.pth/
        └── Swin-Transformer-Object-Detection/
            ├── configs/
            ├── tools/
            ├── data/
            │    └── nuclei/
            │        ├── test/
            │        ├── train/
            │        └── val/
            ├── setup.py
            └── ...
    ```

6. Execute every cell in `inference.ipynb` in `src/` under `swintsfm` kernel
7. You will find the submission file `answer.json` in `src/`
