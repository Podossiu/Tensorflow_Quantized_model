import tensorflow as tf
import pathlib

import os
from tqdm import tqdm
from glob import glob
import random
import numpy as np
import tensorflow_datasets as tfds
import math
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

image_size = 160
size = int(math.floor(image_size / 0.875))
input_shape = (160, 160, 3)
mean = [ 0.485, 0.456, 0.406]
std = [0.229 ** 2, 0.224 ** 2, 0.255 ** 2]
data_augmentation = tf.keras.Sequential(
    [ 
        tf.keras.layers.Resizing(size, size),
        tf.keras.layers.CenterCrop(image_size, image_size),
        tf.keras.layers.Rescaling(1.0/255),
        tf.keras.layers.Normalization(axis = -1, mean = mean,  variance = std),
    ]
)
def get_validation():
    IMAGENET_DATASET_DIR = "/data/imagenet"
    builder = tfds.ImageFolder(IMAGENET_DATASET_DIR)
    ds = builder.as_dataset(split='val', batch_size = 64)

    for idx, batch in enumerate(ds):
        test_x, test_y = batch["image"], batch["label"]
        test_x = data_augmentation(test_x)
        if idx > 0: 
            break
        print(test_x)


        
        
IMAGENET_DATA_DIR = '/data/imagenet/'

#tflite_model_dir = pathlib.Path('./tflite_models')
#tflite_model_dir.mkdir(exist_ok = True, parents = True)

#tflite_model_quant_file = tflite_model_dir/"tflite_model.tflite"

#interpreter = tf.lite.Interpreter(model_path = str(tflite_model_quant_file))
#interpreter.allocate_tensors()

get_validation()
