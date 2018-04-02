import logging
import os
import math

import numpy as np
import matplotlib.pyplot as plt

from autolab_core import Point
from perception import CameraIntrinsics, DepthImage
from gqcnn import Visualizer as vis, Grasp2D

DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/dexnet_2_fc_no_rot_02_13_18_mini'
CAMERA_INTR_DIR =  '/nfs/diskstation/calib/phoxi/phoxi.intr'
IM_FILE_TEMPLATE = 'depth_ims_tf_table'
POSE_FILE_TEMPLATE = 'hand_poses'
METRIC_FILE_TEMPLATE = 'robust_ferrari_canny'
NUM_VIS_SAMPLES = 10
METRIC_THRESH = 0.1
ONLY_POSITIVES = 0
NUM_BINS = 16
GRIPPER_WIDTH = 0.05
PI = math.pi
RAD_TO_DEG = 180 / PI
depth = lambda x: x[:, 2]
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
 
# get im shape
IM_SHAPE = np.load(os.path.join(DATASET_DIR, im_filenames[0]))['arr_0'].shape[1:]
POSE_DIM = np.load(os.path.join(DATASET_DIR, pose_filenames[0]))['arr_0'].shape[1:]

# sample NUM_TEST_SAMPLES files
logging.info('Sampling Test Images')
images = np.zeros((NUM_VIS_SAMPLES,) + IM_SHAPE)
poses = np.zeros((NUM_VIS_SAMPLES,) + POSE_DIM)
metrics = np.zeros((NUM_VIS_SAMPLES,))
counter = 0
while counter < NUM_VIS_SAMPLES:
    file_num = np.random.randint(len(im_filenames))
    im_data = np.load(os.path.join(DATASET_DIR, im_filenames[file_num]))['arr_0']
    pose_data = np.load(os.path.join(DATASET_DIR, pose_filenames[file_num]))['arr_0']
    metric_data = np.load(os.path.join(DATASET_DIR, metric_filenames[file_num]))['arr_0']
    index = np.random.randint(im_data.shape[0])
    im = im_data[index, ...]
    pose = pose_data[index, ...]
    metric = metric_data[index]
    if ONLY_POSITIVES and metric < METRIC_THRESH:
        continue
    images[counter, ...] = im
    poses[counter, ...] = pose
    metrics[counter] = metric
    counter += 1

# process angles and bins
raw_angles = angle(poses)
pi = math.pi
pi_2 = math.pi / 2

neg_ind = np.where(raw_angles < 0)
transform_angles = np.abs(raw_angles) % pi
transform_angles[neg_ind] *= -1
g_90 = np.where(transform_angles > pi_2)
l_neg_90 = np.where(transform_angles < (-1 * pi_2))
transform_angles[g_90] -= pi
transform_angles[l_neg_90] += pi
transform_angles *= -1 # hack to fix reverse angle convention
transform_angles += pi_2
bin_width = pi / NUM_BINS
bins = (transform_angles // bin_width).astype(int)

# visualize
depths = depth(poses)
center = Point(np.asarray([images.shape[1] / 2, images.shape[2] / 2]))
camera_intr = CameraIntrinsics.load(CAMERA_INTR_DIR).resize(0.25)
vis.figure((10, 10))
for i in range(NUM_VIS_SAMPLES):
    logging.info('Visualizing sample: {}'.format(i))
    logging.info('Metric: {}, Depth: {}, Raw Angle: {}, Tranformed Angle: {}, Bin: {}'.format(metrics[i], depths[i], raw_angles[i] * RAD_TO_DEG, transform_angles[i] * RAD_TO_DEG, bins[i]))
    vis.clf()
    vis.imshow(DepthImage(images[i]))
    grasp = Grasp2D(center, raw_angles[i], depths[i], width=GRIPPER_WIDTH, camera_intr=camera_intr)
    vis.grasp(grasp, scale=0.5, show_axis=True, color=plt.cm.RdYlGn(metrics[i]))
    plt.annotate(str(metrics[i]), xy=(1, 1), color='r', size=14)
    vis.show()
