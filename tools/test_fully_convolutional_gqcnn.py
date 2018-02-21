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
# DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/yumi/case_00/phoxi'
DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/mini_dexnet_generic_12_08_17/tensors'
NUM_TEST_SAMPLES = 1
IM_FILE_TEMPLATE = 'depth_ims_tf_table'
POSE_FILE_TEMPLATE = 'hand_poses'
# NUM_CROPS = 494*364
NUM_CROPS = 100
CROP_W = 46
CROP_STRIDE = 2
# FULLY_CONV_CONFIG = {'im_width': 1032, 'im_height': 772}
FULLY_CONV_CONFIG = {'im_width': 64, 'im_height': 64}
# IM_SHAPE = (772, 1032, 1)
IM_SHAPE = (64, 64, 1)
POSE_SHAPE = (1,)
VIS = 1
VIS_POSE_LOC = (5, 5)
FIG_SIZE = (15, 15)
SAVE = 0
SAVE_DIR = '/home/vsatish/Data/dexnet/data/analyses/misc_analyses/fully_conv_gqcnn_test/'
TEST_PAD = 0
pose_parser = lambda p: p[2:3]

# build fully conv network
fully_conv_gqcnn = get_gqcnn_model().load(MODEL_DIR, fully_conv_config=FULLY_CONV_CONFIG)

# get all filenames
all_filenames = os.listdir(DATASET_DIR)
im_filenames = [f for f in all_filenames if f.find(IM_FILE_TEMPLATE) > -1]
pose_filenames = [f for f in all_filenames if f.find(POSE_FILE_TEMPLATE) > -1]
im_filenames.sort(key=lambda x: int(x[-9:-4]))
pose_filenames.sort(key=lambda x: int(x[-9:-4]))

# sample NUM_TEST_SAMPLES files
images = np.zeros((NUM_TEST_SAMPLES,) + IM_SHAPE)
poses = np.zeros((NUM_TEST_SAMPLES,) + POSE_SHAPE)
counter = 0
while counter < NUM_TEST_SAMPLES:
#     file_num = np.random.randint(len(im_filenames))
    file_num = 0
    im_data = np.load(os.path.join(DATASET_DIR, im_filenames[file_num]))['arr_0']
    im = im_data[0]
    rot_mat = cv2.getRotationMatrix2D(((im.shape[0] - 1) / 2, (im.shape[1] - 1) / 2), 95, 1)
    rot_im = cv2.warpAffine(im, rot_mat, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    im_data[0, ..., 0] = rot_im
    pose_data = np.load(os.path.join(DATASET_DIR, pose_filenames[file_num]))['arr_0']
#     index = np.random.randint(im_data.shape[0])
    index = 0
    images[counter, ...] = im_data[index, ...]
    poses[counter, ...] = pose_parser(pose_data[index])
    counter += 1
poses = np.asarray([[0.51]])
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
    fully_conv_pred = fully_conv_gqcnn.predict(images, poses)
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
            if normal_gqcnn_pred[i * NUM_CROPS + x, -1] > 0.1:
                plt.scatter(centers[x, 0], centers[x, 1])
                plt.annotate(str(round(normal_gqcnn_pred[i * NUM_CROPS + x, -1], 1)), xy=centers[x, ::-1], color='r')
            normal_pred_buffer[i * NUM_CROPS + x] = normal_gqcnn_pred[i * NUM_CROPS + x, -1]
#        plt.scatter(centers[:, 1], centers[:, 0])
        plt.annotate(str(poses[i, 0]), xy=VIS_POSE_LOC, color='r', size=14)
#        plt.tight_layout()
        if VIS:
            plt.show()
        if SAVE:
            num_tag = format(i, '0{}'.format(decimal_places))
            filename = 'normal_gqcnn_{}.png'.format(num_tag)
            plt.savefig(os.path.join(SAVE_DIR, filename))

        plt.figure(figsize=FIG_SIZE)
        plt.title('Fully-Convolutional GQCNN')
        plt.imshow(images[i, ..., 0], cmap='gray')
        output_dim_h = fully_conv_pred.shape[1]
        output_dim_w = fully_conv_pred.shape[2]
        for x in range(NUM_CROPS):
            try:
                if fully_conv_pred[i, x // output_dim_w, x % output_dim_w, -1] > 0.2:
                    plt.annotate(str(round(fully_conv_pred[i, x // output_dim_w, x % output_dim_w, -1], 1)), xy=centers[x, ::-1], color='r')
            except:
                import IPython
                IPython.embed()  
            fully_conv_pred_buffer[i * NUM_CROPS + x] = fully_conv_pred[i, x // output_dim_w, x % output_dim_w, -1]
#        plt.scatter(centers[:, 1], centers[:, 0])
        plt.annotate(str(poses[i, 0]), xy=VIS_POSE_LOC, color='r', size=14)
        plt.tight_layout()
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
