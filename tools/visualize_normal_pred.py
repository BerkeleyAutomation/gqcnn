import logging
import os
import math

import numpy as np
import matplotlib.pyplot as plt

from autolab_core import Point
from perception import CameraIntrinsics, DepthImage
from gqcnn.model import get_gqcnn_model
from gqcnn import Visualizer as vis, Grasp2D

# MODEL_DIR = '/home/vsatish/Data/dexnet/data/models/test_dump/model_ebhsmdqmjd'
MODEL_DIR = '/home/vsatish/Data/dexnet/data/models/test_dump/model_ldjvttndkj/'
# DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/dexnet_2_fcn'
DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/salt_cube_leg_04_23_18/'
CAMERA_INTR_DIR =  '/nfs/diskstation/calib/phoxi/phoxi.intr'
IM_FILE_TEMPLATE = 'depth_ims_tf_table'
POSE_FILE_TEMPLATE = 'hand_poses'
METRIC_FILE_TEMPLATE = 'robust_wrench_resistance'
NUM_TEST_SAMPLES = 10
POSE_DIM = 1
ONLY_POSITIVES = 1
# POS_THRESH = 0.1
POS_THRESH = 0.75
pose_parser = lambda p: p[2:3]
GRIPPER_WIDTH = 0.05
DEBUG = 1
SEED = 12545

# set random seed if debugging
if DEBUG:
    np.random.seed(seed=SEED)

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

# get im shape
IM_SHAPE = np.load(os.path.join(DATASET_DIR, im_filenames[0]))['arr_0'].shape[1:]

# sample NUM_TEST_SAMPLES files
logging.info('Sampling Test Images')
images = np.zeros((NUM_TEST_SAMPLES,) + IM_SHAPE)
poses = np.zeros((NUM_TEST_SAMPLES,) + (POSE_DIM,))
metrics = np.zeros((NUM_TEST_SAMPLES,))
counter = 0
while counter < NUM_TEST_SAMPLES:
    file_num = np.random.randint(len(im_filenames))
    im_data = np.load(os.path.join(DATASET_DIR, im_filenames[file_num]))['arr_0']
    pose_data = np.load(os.path.join(DATASET_DIR, pose_filenames[file_num]))['arr_0']
    metric_data = np.load(os.path.join(DATASET_DIR, metric_filenames[file_num]))['arr_0']
    index = np.random.randint(im_data.shape[0])
    im = im_data[index, ...]
    pose = pose_data[index, ...]
    metric = metric_data[index]
    if ONLY_POSITIVES and metric < POS_THRESH:
        continue
    images[counter, ...] = im
    poses[counter, ...] = pose_parser(pose)
    metrics[counter] = metric
    counter += 1

poses = np.tile(np.asarray([[0.2]]), (NUM_TEST_SAMPLES, 1))

# load Angular-GQCNN model
logging.info('Loading GQCNN')
gqcnn = get_gqcnn_model().load(MODEL_DIR)

# predict test images
logging.info('Inferring')
gqcnn.open_session()
preds = gqcnn.predict(images, poses)
gqcnn.close_session()

# generate grasps
logging.info('Generating grasp visualizations')
grasps = []
center = Point(np.asarray([images.shape[1] / 2, images.shape[2] / 2]))
camera_intr = CameraIntrinsics.load(CAMERA_INTR_DIR).resize(0.25)
for i in range(NUM_TEST_SAMPLES):
    grasps.append(Grasp2D(center, 0.0, poses[i, 0], width=GRIPPER_WIDTH, camera_intr=camera_intr))

# visualize
logging.info('Beginning Visualization')
vis.figure(size=(10, 10))
for i in range(NUM_TEST_SAMPLES):
    logging.info('Depth: {}'.format(poses[i, 0]))
    logging.info('Pred: {}'.format(preds[i]))
    vis.clf()
    vis.imshow(DepthImage(images[i]))
    vis.grasp(grasps[i], scale=0.5, show_axis=True, color=plt.cm.RdYlGn(preds[i, 1]))
    plt.annotate(str(metrics[i]), xy=(1, 1), color='r', size=14)
    vis.show()    
