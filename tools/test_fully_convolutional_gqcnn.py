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
import math
import cv2
import skimage.transform as skt
import sys

from autolab_core import YamlConfig
from gqcnn.model import get_gqcnn_model

MODEL_DIR = '/home/vsatish/Data/dexnet/data/models/test_dump/model_ebhsmdqmjd'
#FULLY_CONV_CONFIG = 'cfg/tools/fully_conv.yaml'

# restart and setup logger
logging.shutdown()
reload(logging)
logging.getLogger().setLevel(logging.INFO)

# build normal network
normal_gqcnn = get_gqcnn_model().load(MODEL_DIR)

####################### EQUIVALENCE CHECK ON SINGLE STRIDE #######################
DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/dexnet_2_fcn'
NUM_TEST_SAMPLES = 10
IM_FILENAME = 'depth_ims_tf_table_00000.npz'
POSE_FILENAME = 'hand_poses_00000.npz'
FULLY_CONV_CONFIG = {'im_width': 46, 'im_height': 46}

# build fully conv network
_fully_conv_gqcnn = get_gqcnn_model().load(MODEL_DIR, fully_conv_config=FULLY_CONV_CONFIG)

# load test samples
images = np.load(os.path.join(DATASET_DIR, IM_FILENAME))['arr_0'][:NUM_TEST_SAMPLES]
poses = np.load(os.path.join(DATASET_DIR, POSE_FILENAME))['arr_0'][:NUM_TEST_SAMPLES, 2:3]

# predict
normal_gqcnn.open_session()
_fully_conv_gqcnn.open_session()
normal_pred = normal_gqcnn.predict(images, poses)
fully_conv_pred = _fully_conv_gqcnn.predict(images, poses)
_fully_conv_gqcnn.close_session()

# cross-reference outputs
assert (np.count_nonzero(np.squeeze(fully_conv_pred) - normal_pred) == 0), 'SANITY CHECK ON SINGLE STRIDE FAILED!'

####################### TEST MULTIPLE STRIDES #######################
DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/yumi/case_00/phoxi'
# DATASET_DIR = '/nfs/diskstation/projects/dex-net/parallel_jaws/datasets/dexnet_4/phoxi_v7/images/tensors/'
NUM_TEST_SAMPLES = 1
# NUM_TEST_SAMPLES = 10
IM_FILE_TEMPLATE = 'depth_ims_tf_table'
# IM_FILE_TEMPLATE = 'depth_ims'
POSE_FILE_TEMPLATE = 'hand_poses'
NUM_CROPS = 107*75
# NUM_CROPS = 378*278
CROP_W = 46
CROP_STRIDE = 2
FULLY_CONV_CONFIG = {'im_width': 258, 'im_height': 193}
# FULLY_CONV_CONFIG = {'im_width': 800, 'im_height': 600}
IM_SHAPE = (193, 258, 1)
# IM_SHAPE = (600, 800, 1)
POSE_SHAPE = (1,)
VIS = 1
VIS_POSE_LOC = (5, 5)
FIG_SIZE = (15, 15)
SAVE = 0
SAVE_DIR = '/home/vsatish/Data/dexnet/data/analyses/misc_analyses/fully_conv_gqcnn_test/'
TEST_PAD = 0
pose_parser = lambda p: p[2:3]

# build fully conv network
fully_conv_gqcnn = get_gqcnn_model().load(MODEL_DIR, fully_conv_config=FULLY_CONV_CONFIG, conv_filt_rot=20)

# get all filenames
all_filenames = os.listdir(DATASET_DIR)
im_filenames = [f for f in all_filenames if f.find(IM_FILE_TEMPLATE) > -1]
pose_filenames = [f for f in all_filenames if f.find(POSE_FILE_TEMPLATE) > -1]
im_filenames.sort(key=lambda x: int(x[-9:-4]))
pose_filenames.sort(key=lambda x: int(x[-9:-4]))

# sample NUM_TEST_SAMPLES files
images = np.zeros((NUM_TEST_SAMPLES,) + IM_SHAPE)
un_rot_images = np.zeros((NUM_TEST_SAMPLES,) + IM_SHAPE)
poses = np.zeros((NUM_TEST_SAMPLES,) + POSE_SHAPE)
counter = 0
while counter < NUM_TEST_SAMPLES:
#     file_num = np.random.randint(len(im_filenames))
    file_num = 0
    im_data = np.load(os.path.join(DATASET_DIR, im_filenames[file_num]))['arr_0']
    index = 0
    im = im_data[index, ..., 0]
    im[np.where(im > 1.0)] = 1.0
    im = skt.rescale(im, 0.25)
    un_rot_images[counter, ..., 0] = im
    rot_mat = cv2.getRotationMatrix2D(((im.shape[1] - 1) / 2, (im.shape[0] - 1) / 2), 90, 1)
    rot_im = cv2.warpAffine(im, rot_mat, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    im_data = np.zeros((1, 193, 258, 1))
    im_data[index, ..., 0] = rot_im
#     pose_data = np.load(os.path.join(DATASET_DIR, pose_filenames[file_num]))['arr_0']
#     index = np.random.randint(im_data.shape[0])
    index = index
    images[counter, ...] = im_data[index, ...]
#    poses[counter, ...] = pose_parser(pose_data[index])
    counter += 1
#poses = np.asarray([[0.85], [0.86], [0.87], [0.88], [0.87], [0.86], [0.85], [0.88], [0.87], [0.88]])
poses = np.asarray([[0.8]])
# first benchmark cropping + pass through normal GQCNN
start_time = time.time()
crop_images = np.zeros((images.shape[0]*NUM_CROPS, CROP_W, CROP_W, images.shape[3]))
crop_poses = np.zeros((images.shape[0]*NUM_CROPS,) + POSE_SHAPE)
centers = np.zeros((NUM_CROPS, 2))
calc_centers = True
for i in range(images.shape[0]):
    im = images[i]
    width = images.shape[2]
    height = images.shape[1]
    index = 0
    for h in range(0, height - CROP_W + CROP_STRIDE, CROP_STRIDE):
        for w in range(0, width - CROP_W + CROP_STRIDE, CROP_STRIDE):
            crop = im[h:h + CROP_W, w:w + CROP_W, ...]
            if (crop.shape[0] != 46):
                crop_images[i * NUM_CROPS + index] = np.ones((46, 46, 1))
            else:                
                crop_images[i * NUM_CROPS + index] = im[h:h + CROP_W, w:w + CROP_W, ...]
            if calc_centers:
                centers[index] = np.asarray([h + CROP_W / 2, w + CROP_W / 2])
            index += 1
    calc_centers = False
    crop_poses[i * NUM_CROPS:i * NUM_CROPS + NUM_CROPS] = poses[i]
crop_time = time.time() - start_time

start_time = time.time()
normal_gqcnn_pred = normal_gqcnn.predict(crop_images, crop_poses)
eval_time = time.time() - start_time

# next benchmark strided pass through fully-convolutional GQCNN
fully_conv_gqcnn.open_session()
start_time = time.time()
if TEST_PAD:
    padded_ims = np.zeros((images.shape[0]*NUM_CROPS,) + images.shape[1:])
    padded_ims[:, :CROP_W, :CROP_W, ...] = crop_images
    pred = fully_conv_gqcnn.predict(padded_ims, crop_poses)
    num_crop_sq = int(np.sqrt(NUM_CROPS))
    fully_conv_pred = np.zeros((NUM_TEST_SAMPLES, num_crop_sq, num_crop_sq, 2))
    for i in range(images.shape[0]):
        for y in range(num_crop_sq):
            for x in range(num_crop_sq):
                fully_conv_pred[i, y, x, ...] = pred[NUM_CROPS * i + int(math.sqrt(NUM_CROPS)) * y + x, y, x, ...] 
else: 
    fully_conv_pred = fully_conv_gqcnn.predict(un_rot_images, poses)
fully_conv_duration = time.time() - start_time

normal_pred_buffer = np.zeros((NUM_TEST_SAMPLES * NUM_CROPS,))
fully_conv_pred_buffer = np.zeros((NUM_TEST_SAMPLES * NUM_CROPS,))
decimal_places = len(str(images.shape[0]))
if VIS or SAVE:
    for i in range(images.shape[0]):
        plt.figure()
        plt.title('Normal GQCNN')
        plt.imshow(images[i, ..., 0], cmap='gray')
        for x in range(NUM_CROPS):
            if normal_gqcnn_pred[i * NUM_CROPS + x, -1] > 0.2:
                plt.scatter(centers[x, 1], centers[x, 0], color=plt.cm.RdYlGn(normal_gqcnn_pred[i * NUM_CROPS + x, -1]))
#                plt.annotate(str(round(normal_gqcnn_pred[i * NUM_CROPS + x, -1], 1)), xy=centers[x, ::-1], color='r')
            normal_pred_buffer[i * NUM_CROPS + x] = normal_gqcnn_pred[i * NUM_CROPS + x, -1]
#        plt.scatter(centers[:, 1], centers[:, 0], color=plt.cm.RdYlGn(normal_pred_buffer[i * NUM_CROPS:i * NUM_CROPS + NUM_CROPS]))
        plt.annotate(str(poses[i, 0]), xy=VIS_POSE_LOC, color='r', size=14)
#        plt.tight_layout()
        if VIS:
            plt.show()
        if SAVE:
            num_tag = format(i, '0{}'.format(decimal_places))
            filename = 'normal_gqcnn_{}.png'.format(num_tag)
            plt.savefig(os.path.join(SAVE_DIR, filename))
   
        plt.figure()
        plt.title('Fully-Convolutional GQCNN')
        plt.imshow(un_rot_images[i, ..., 0], cmap='gray')
        output_dim_h = fully_conv_pred.shape[1]
        output_dim_w = fully_conv_pred.shape[2]
        for x in range(NUM_CROPS):
            if fully_conv_pred[i, x // output_dim_w, x % output_dim_w, -1] > 0.2:
                plt.scatter(centers[x, 1], centers[x, 0], color=plt.cm.RdYlGn(fully_conv_pred[i, x // output_dim_w, x % output_dim_w, -1]))
#                    plt.annotate(str(round(fully_conv_pred[i, x // output_dim_w, x % output_dim_w, -1], 1)), xy=centers[x, ::-1], color='r')
            fully_conv_pred_buffer[i * NUM_CROPS + x] = fully_conv_pred[i, x // output_dim_w, x % output_dim_w, -1]
#        plt.scatter(centers[:, 1], centers[:, 0])
        plt.annotate(str(poses[i, 0]), xy=VIS_POSE_LOC, color='r', size=14)
#        plt.tight_layout()
        if VIS:
            plt.show()
        if SAVE:
            num_tag = format(i, '0{}'.format(decimal_places))
            filename = 'fully_conv_gqcnn_{}.png'.format(num_tag)
            plt.savefig(os.path.join(SAVE_DIR, filename))
    
# compare benchmark times of both GQCNNs
logging.info('Normal GQCNN took: {} seconds while Fully-Convolutional GQCNN took: {} seconds. Crop time was: {}, Evaluation time was: {}'.format(crop_time + eval_time, fully_conv_duration, crop_time, eval_time))

# close TF sessions
normal_gqcnn.close_session()
fully_conv_gqcnn.close_session()

logging.info('Test Finished')
