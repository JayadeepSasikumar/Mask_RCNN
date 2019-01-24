import cv2
import math
# import matplotlib
# import matplotlib.pyplot as plt
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
# from mrcnn import visualize
from mrcnn.model import log

# %matplotlib inline

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
    to the DAVIS 2016 dataset.
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

    # Keeping the STEPS_PER_EPOCH the same as number of training images
    STEPS_PER_EPOCH = 2079

    # Keeping the VALIDATION_STEPS the same as number of val images
    VALIDATION_STEPS = 1054
    

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
        image_classes = os.listdir(images_dir)
        if '.DS_Store' in image_classes:
          image_classes.remove('.DS_Store')
        for i, image_class in enumerate(image_classes):
            self.add_class("davis", i + 1, image_class)
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
                           pic_name=pic_name, pic_class=pic_class,
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

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

# Training the head.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=50, 
            layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=100, 
            layers="all")

model_path = os.path.join(MODEL_DIR, "mask_rcnn_davis.h5")
model.keras_model.save_weights(model_path)
