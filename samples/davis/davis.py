import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import re
import sys
import time

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

DATA_DIR = os.path.join(ROOT_DIR, "davis_data")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class DAVISConfig(Config):
    """Configuration for training on the DAVIS 2017 dataset.
    Derives from the base Config class and overrides values specific
    to the DAVIS 2017 dataset.
    """
    # Give the configuration a recognizable name
    NAME = "davis"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + object

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 832
    IMAGE_MAX_DIM = 832
    IMAGE_RESIZE_MODE = "square"  # This is the default option as well

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 5  #32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 2079

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 1054
    
config = DAVISConfig()
    

config = DAVISConfig()


class DAVISDataset(utils.Dataset):
    """Encapsulates the DAVIS dataset.
    """
    def image_reference(self, image_id):
        """Return the davis data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "davis":
            return info["davis"]
        else:
            super(self.__class__, self).image_reference(image_id)
    
    def load_images(self, mode='train'):
        """
        Loads the 480p images from the DAVIS dataset.
        """
        images_dir = os.path.join(DATA_DIR, 'JPEGImages', '480p')
#         image_classes = os.listdir(images_dir)
#         if '.DS_Store' in image_classes:
#           image_classes.remove('.DS_Store')
#         for i, image_class in enumerate(image_classes):
#             self.add_class("davis", i + 1, image_class)
        self.add_class('davis', 1, 'object')
        image_paths_file_name = mode + '.txt'
        image_paths_file_path = os.path.join(DATA_DIR, image_paths_file_name)
        with open(image_paths_file_path, 'r') as image_paths_file:
            for i, line in enumerate(image_paths_file):
                try:
                  image_path, mask_path = line.split()[0], line.split()[1]
                except IndexError:
                  continue
                image_path = DATA_DIR + image_path
                mask_path = DATA_DIR + mask_path
                pic_name = image_path.split('/')[-1]
                pic_class = image_path.split('/')[-2]
                self.add_image("davis", image_id=i, path=image_path,
                           pic_name=pic_name, pic_class='object',
                           mask_path=mask_path)
            
    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        mask_path = info['mask_path']
        pic_class = info['pic_class']
        mask = cv2.imread(mask_path)
        mask = mask[:, :, 0:1]
        class_ids = np.array([self.class_names.index(pic_class)])
        return mask.astype(np.bool), class_ids.astype(np.int32)

# Training dataset
dataset_train = DAVISDataset()
dataset_train.load_images(mode='train')
dataset_train.prepare()

# Validation dataset
dataset_val = DAVISDataset()
dataset_val.load_images(mode='val')
dataset_val.prepare()

# Test dataset
dataset_test = DAVISDataset()
dataset_test.load_images(mode='test')
dataset_test.prepare()


# class InferenceConfig(DAVISConfig):
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#     DETECTION_MIN_CONFIDENCE = 0.8

# inference_config = InferenceConfig()

# # Recreate the model in inference mode
# model = modellib.MaskRCNN(mode="inference", 
#                           config=inference_config,
#                           model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
# model_path = os.path.join('/Users/jayadeepsasikumar/mask_rcnn/Mask_RCNN/logs/davis20181101T1232',
#                           'mask_rcnn_davis_0001.h5')
# model.load_weights(model_path, by_name=True)

# predicted_masks = {}
# for image_id in dataset_test.image_ids:
#     image, image_meta, gt_class_id, gt_bbox, gt_mask =\
#         modellib.load_image_gt(dataset_val, inference_config,
#                                image_id, use_mini_mask=False)
#     molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
#     # Run object detection
#     results = model.detect([image], verbose=0)
#     result = results[0]
# #     predicted_masks[image_id] = result['masks'][0]

# with open('test_predictions.pickle', 'wb') as fp:
#     pickle.dump(predicted_masks, fp)