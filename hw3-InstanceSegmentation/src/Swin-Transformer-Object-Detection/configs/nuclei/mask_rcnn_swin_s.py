# The new config inherits a base config to highlight the necessary
# modification
_base_ = '../swin/mask_rcnn_swin_small_patch4_window7_mstrain_'\
         '480-800_adamw_3x_coco.py'

# We also need to change the num_classes in head to match the dataset's
# annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

runner = dict(type='EpochBasedRunner', max_epochs=600)

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('nuclei',)
data = dict(
    train=dict(
        img_prefix='data/nuclei/train/',
        classes=classes,
        ann_file='data/nuclei/train/annotation.json'),
    val=dict(
        img_prefix='data/nuclei/val/',
        classes=classes,
        ann_file='data/nuclei/val/annotation.json'),
    # val=dict(
    #     img_prefix='data/nuclei/train/',
    #     classes=classes,
    #     ann_file='data/nuclei/train/annotation.json'),
    test=dict(
        img_prefix='data/nuclei/test/',
        classes=classes,
        ann_file='data/nuclei/test/annotation.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/mask_rcnn_swin_small_patch4_window7.pth'
# load_from = 'work_dirs/mask_rcnn_x101_64x4d_fpn_1x_coco/epoch_7.pth'
