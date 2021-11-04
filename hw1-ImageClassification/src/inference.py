import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# Set device and directories
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_DIR = '../2021VRDL_HW1_datasets'
RLS_CKPT_DIR = '../checkpoints/RELEASE'
OUT_DIR = '.'

# Nets to predict testing data
nets = [
    (torchvision.models.resnet152(pretrained=False),
        'resnet152_130e', 0.622156),
    (torchvision.models.resnext101_32x8d(pretrained=False),
        'resnext101_32x8d_advtrsf_160e', 0.679196),
    (torchvision.models.resnext101_32x8d(pretrained=False),
        'resnext101_32x8d_advtrsf_nonoised_250e', 0.661721),
    (torchvision.models.resnext101_32x8d(pretrained=False),
        'resnext101_32x8d_70e', 0.631058),
    (torchvision.models.resnext101_32x8d(pretrained=False),
        'resnext101_32x8d_400e', 0.666667),
]


# Read the classes
with open(f'{DATA_DIR}/classes.txt') as f:
    classes = [x.strip() for x in f.readlines()]


def get_class_display(pred):
    # Returns the class name according to the given label
    return classes[pred]


def main():
    # Initial the nets
    for net, ckpt, _ in nets:
        # Set the size of final layer to 200
        num_features = net.fc.in_features
        net.fc = torch.nn.Linear(num_features, 200)

        # Load the pretrained weughts
        PATH = f'{RLS_CKPT_DIR}/{ckpt}.pth'
        net.load_state_dict(torch.load(PATH))
        print('Load checkpoint from', PATH)

        # Put net to device
        net = net.to(device)

    # Testing
    submission = []

    with torch.no_grad():
        # Data transform
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        with open(f'{DATA_DIR}/testing_img_order.txt') as f:
            # Read all the testing images
            test_image_names = [x.strip() for x in f.readlines()]

        for img_name in test_image_names:
            # image order is important to your result
            img_path = os.path.join(f'{DATA_DIR}/testing_images/', img_name)
            img = transform(Image.open(img_path).convert('RGB'))
            img = img[None, :]
            img = img.to(device)

            outputs_list = torch.zeros(200).to(device)
            for net, _, weight in nets:
                net.eval()
                outputs = net(img)  # the predicted category
                outputs_list = outputs_list + outputs * weight

            _, predicted_class = torch.max(outputs_list, 1)
            predicted_class_display = get_class_display(int(predicted_class))
            submission.append([img_name, predicted_class_display])

    # Save the submission
    np.savetxt(f'{OUT_DIR}/answer.txt', submission, fmt='%s')


if __name__ == '__main__':
    main()
