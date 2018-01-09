"""
Script to test Fully-Convolutional version of GQCNN
Author: Vishal Satish
"""

import os
import logging
import numpy as np
import tensorflow as tf
import IPython as ip
import time
import matplotlib.pyplot as plt

from autolab_core import YamlConfig
from gqcnn import GQCNN

MODEL_DIR = '/home/vsatish/Data/dexnet/data/models/grasp_quality/dev_vishal_models/mini_dexnet_01_06_18/'
#FULLY_CONV_CONFIG = 'cfg/tools/fully_conv.yaml'

# restart and setup logger
logging.shutdown()
reload(logging)
logging.getLogger().setLevel(logging.INFO)

# build normal network
normal_gqcnn = GQCNN.load(MODEL_DIR)

####################### EQUIVALENCE CHECK ON SINGLE STRIDE #######################
DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/mini_dexnet_all_trans_01_20_17/'
NUM_TEST_SAMPLES = 10
IM_FILENAME = 'depth_ims_tf_table_00000.npz'
POSE_FILENAME = 'hand_poses_00000.npz'
FULLY_CONV_CONFIG = {'im_width': 32, 'im_height': 32}

# build fully conv network
_fully_conv_gqcnn = GQCNN.load(MODEL_DIR, fully_conv_config=FULLY_CONV_CONFIG)

# load test samples
images = np.load(os.path.join(DATASET_DIR, IM_FILENAME))['arr_0'][:NUM_TEST_SAMPLES]
poses = np.load(os.path.join(DATASET_DIR, POSE_FILENAME))['arr_0'][:NUM_TEST_SAMPLES, 2:3]

# predict
normal_pred = normal_gqcnn.predict(images, poses)
fully_conv_pred = _fully_conv_gqcnn.predict(images, poses)

# cross-reference outputs
assert (np.count_nonzero(np.squeeze(fully_conv_pred) - normal_pred) == 0), 'SANITY CHECK ON SINGLE STRIDE FAILED!'

####################### TEST MULTIPLE STRIDES #######################
DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/bar_clamp_generic_11_25_17/tensors'
NUM_TEST_SAMPLES = 1
IM_FILENAME = 'depth_ims_tf_table_00000.npz'
POSE_FILENAME = 'hand_poses_00000.npz'
NUM_CROPS = 17*17
CROP_W = 32
CROP_STRIDE = 2
FULLY_CONV_CONFIG = {'im_width': 64, 'im_height': 64}

# build fully conv network
fully_conv_gqcnn = GQCNN.load(MODEL_DIR, fully_conv_config=FULLY_CONV_CONFIG)

# load test samples
images = np.load(os.path.join(DATASET_DIR, IM_FILENAME))['arr_0'][:NUM_TEST_SAMPLES]
poses = np.load(os.path.join(DATASET_DIR, POSE_FILENAME))['arr_0'][:NUM_TEST_SAMPLES, 2:3]

# first benchmark cropping + pass through normal GQCNN
start_time = time.time()
crop_images = np.zeros((images.shape[0]*NUM_CROPS, CROP_W, CROP_W, images.shape[3]))
centers = np.zeros((NUM_CROPS, 2))
for i in range(images.shape[0]):
    im = images[i]
    width = images.shape[1]
    index = 0
    for h in range(0, width - CROP_W + CROP_STRIDE, CROP_STRIDE):
        for w in range(0, width - CROP_W + CROP_STRIDE, CROP_STRIDE):
            crop_images[i * NUM_CROPS + index] = im[h:h + CROP_W, w:w + CROP_W, ...]
            centers[i * NUM_CROPS + index] = np.asarray([h + CROP_W / 2, w + CROP_W / 2])
            index += 1
normal_gqcnn_pred = normal_gqcnn.predict(crop_images, np.tile(poses, (NUM_CROPS, 1)))
normal_gqcnn_duration = time.time() - start_time

# next benchmark strided pass through fully-convolutional GQCNN
start_time = time.time()
fully_conv_pred = fully_conv_gqcnn.predict(images, poses)
fully_conv_duration = time.time() - start_time

# plot predictions of both models
plt.figure()
plt.title('Normal GQCNN')
plt.imshow(images[0, :, :, 0], cmap='gray')
for x in range(centers.shape[0]):
    plt.annotate(str(round(normal_gqcnn_pred[x, 0], 1)), xy=centers[x, ::-1])
    index += 1
plt.scatter(centers[:, 0], centers[:, 1])
plt.show()

plt.figure()
plt.title('Fully-Convolutional GQCNN')
plt.imshow(images[0, :, :, 0], cmap='gray')
output_dim = fully_conv_pred.shape[1]
for x in range(centers.shape[0]):
    plt.annotate(str(round(fully_conv_pred[0, x / output_dim, x % output_dim, 0], 1)), xy=centers[x, ::-1])
    index += 1
    plt.scatter(centers[:, 0], centers[:, 1])
plt.show()

# compare benchmark times of both GQCNNs
logging.info('Normal GQCNN took: {} seconds while Fully-Convolutional GQCNN took: {} seconds.'.format(normal_gqcnn_duration, fully_conv_duration))

logging.info('Test Finished')
