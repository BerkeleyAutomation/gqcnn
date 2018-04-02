import os
import math
import cPickle as pkl
import logging

import numpy as np

from gqcnn.utils.learning_analysis import ClassificationResult
from gqcnn.model import get_gqcnn_model

DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/dexnet_2_fc_no_rot_02_13_18_mini'
# DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/dexnet_2_fcn_mini'
MODEL_DIR = '/home/vsatish/Data/dexnet/data/models/test_dump/model_jgvkqlgazn'
# MODEL_DIR = '/home/vsatish/Data/dexnet/data/models/grasp_quality/dev_vishal_models/mini_dexnet_02_07_18_tf'
IM_FILE_TEMPLATE = 'depth_ims_tf_table'
POSE_FILE_TEMPLATE = 'hand_poses'
METRIC_FILE_TEMPLATE = 'robust_ferrari_canny'
METRIC_THRESH = 0.1
# METRIC_THRESH = 0.002
TRAIN_INDEX_FNAME = 'train_indices_image_wise.pkl'
VAL_INDEX_FNAME = 'val_indices_image_wise.pkl'
NUM_BINS = 16
# NUM_BINS = 1
PI = math.pi
PI_2 = PI / 2
depth = lambda x: x[:, 2:3]
angle = lambda x: x[:, 3]

# restart and setup logger
logging.shutdown()
reload(logging)
logging.getLogger().setLevel(logging.INFO)
 
# get all filenames
logging.info('Reading Filenames')
all_filenames = os.listdir(DATASET_DIR)
im_filenames = [f for f in all_filenames if f.find(IM_FILE_TEMPLATE) > -1]
pose_filenames = [f for f in all_filenames if f.find(POSE_FILE_TEMPLATE) > -1]
metric_filenames = [f for f in all_filenames if f.find(METRIC_FILE_TEMPLATE) > -1]
im_filenames.sort(key=lambda x: int(x[-9:-4]))
pose_filenames.sort(key=lambda x: int(x[-9:-4]))
metric_filenames.sort(key=lambda x: int(x[-9:-4]))

# load training and validation indices
logging.info('Loading Indices')
with open(os.path.join(MODEL_DIR, TRAIN_INDEX_FNAME), 'rb') as fhandle:
    train_index_map = pkl.load(fhandle)
with open(os.path.join(MODEL_DIR, VAL_INDEX_FNAME), 'rb') as fhandle:
    val_index_map = pkl.load(fhandle)

# load gqcnn model
logging.info('Loading GQ-CNN Model')
gqcnn = get_gqcnn_model().load(MODEL_DIR)
gqcnn.open_session()

# compute avg training and validation error
train_errors = np.zeros((len(im_filenames),))
val_errors = np.zeros((len(im_filenames),))
for i in range(len(im_filenames)):
    images = np.load(os.path.join(DATASET_DIR, im_filenames[i]))['arr_0']
    poses = np.load(os.path.join(DATASET_DIR, pose_filenames[i]))['arr_0']
    metrics = np.load(os.path.join(DATASET_DIR, metric_filenames[i]))['arr_0']

    train_indices = train_index_map[im_filenames[i]]
    val_indices = val_index_map[im_filenames[i]]

    logging.info('Predicting file {}'.format(i))
    preds = gqcnn.predict(images, depth(poses))
    train_pred = preds[train_indices]
    val_pred = preds[val_indices]

    labels = (1 * (metrics > METRIC_THRESH)).astype(np.uint8)
    raw_angles = angle(poses)

    neg_ind = np.where(raw_angles < 0)
    transform_angles = np.abs(raw_angles) % PI
    transform_angles[neg_ind] *= -1
    g_90 = np.where(transform_angles > PI_2)
    l_neg_90 = np.where(transform_angles < (-1 * PI_2))
    transform_angles[g_90] -= PI
    transform_angles[l_neg_90] += PI
    transform_angles *= -1 # hack to fix reverse angle convention
    transform_angles += PI_2
    bin_width = PI / NUM_BINS

    pred_mask = np.zeros((images.shape[0], NUM_BINS * 2))
    for j in range(transform_angles.shape[0]):
        pred_mask[j, int((transform_angles[j] // bin_width)*2)] = True
        pred_mask[j, int((transform_angles[j] // bin_width)*2 + 1)] = True

#     import IPython
#     IPython.embed()
    train_errors[i] = ClassificationResult([train_pred[pred_mask[train_indices].astype(np.bool)].reshape((-1, 2))], [labels[train_indices]]).error_rate 
    val_errors[i] = ClassificationResult([val_pred[pred_mask[val_indices].astype(bool)].reshape((-1, 2))], [labels[val_indices]]).error_rate
 
    logging.info('Training Error: {}, Validation Error: {}'.format(train_errors[i], val_errors[i]))

logging.info('Average Training Error: {}, Average Validation Error: {}'.format(np.mean(train_errors), np.mean(val_errors)))

