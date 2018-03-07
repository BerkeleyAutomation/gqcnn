import logging
import os
import math

import numpy as np
import matplotlib.pyplot as plt

from autolab_core import Point
from perception import CameraIntrinsics, DepthImage
from gqcnn.model import get_gqcnn_model
from gqcnn import Visualizer as vis, Grasp2D

MODEL_DIR = '/home/vsatish/Data/dexnet/data/models/test_dump/model_ljzryvduzl'
DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/dexnet_2_fc_no_rot_02_13_18_mini'
CAMERA_INTR_DIR =  '/nfs/diskstation/calib/phoxi/phoxi.intr'
IM_FILE_TEMPLATE = 'depth_ims_tf_table'
POSE_FILE_TEMPLATE = 'hand_poses'
METRIC_FILE_TEMPLATE = 'robust_ferrari_canny'
NUM_TEST_SAMPLES = 10
POSE_DIM = 1
NUM_BINS = 16
pose_parser = lambda p: p[2:3]

CAM_NAME = 'phoxi_overhead'
FX = 520
FY = 520
CX = 299.5
CY = 299.5
WIDTH = 600
HEIGHT = 600

GRIPPER_WIDTH = 0.05

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
ground_truth_angles = np.zeros((NUM_TEST_SAMPLES,))
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
    if metric < 0.1:
        continue
    images[counter, ...] = im
    poses[counter, ...] = pose_parser(pose)
    ground_truth_angles[counter] = pose[3]
    metrics[counter] = metric
    counter += 1

# load Angular-GQCNN model
logging.info('Loading Angular-GQCNN')
ang_gqcnn = get_gqcnn_model().load(MODEL_DIR)

# predict test images
logging.info('Inferring')
ang_gqcnn.open_session()
angular_pred = ang_gqcnn.predict(images, poses)
ang_gqcnn.close_session()

# generate grasps
logging.info('Generating grasp visualizations')
grasps = []
bin_width = math.pi / 2 / NUM_BINS
center = Point(np.asarray([images.shape[1] / 2, images.shape[2] / 2]))
#camera_intr = CameraIntrinsics(CAM_NAME, fx=FX, fy=FY, cx=CX, cy=CY, width=WIDTH, height=HEIGHT)
camera_intr = CameraIntrinsics.load(CAMERA_INTR_DIR).resize(0.25)
for i in range(NUM_TEST_SAMPLES):
    sample_grasps = []
    for j in range(NUM_BINS):
        bin_cent_ang = j * bin_width + bin_width / 2
#        logging.info('Bin width: {}, Ang: {}'.format(bin_width, bin_cent_ang))
        sample_grasps.append(Grasp2D(center, bin_cent_ang - math.pi / 4, poses[i, 0], width=GRIPPER_WIDTH, camera_intr=camera_intr))
    sample_grasps.append(Grasp2D(center, ground_truth_angles[i], poses[i, 0], width=GRIPPER_WIDTH, camera_intr=camera_intr))
    grasps.append(sample_grasps)

# visualize
logging.info('Beginning Visualization')
vis.figure(size=(10, 10))
for i in range(NUM_TEST_SAMPLES):
    logging.info('Angular Pred: {}'.format(angular_pred[i]))
    vis.clf()
    vis.imshow(DepthImage(images[i]))
    for j, grasp in enumerate(grasps[i][:-1]):
        vis.grasp(grasp, scale=0.5, show_axis=True, color=plt.cm.RdYlGn(angular_pred[i, j * 2 + 1]))
    vis.grasp(grasps[i][-1], scale=0.5, show_axis=True, color='black')
    plt.annotate(str(metrics[i]), xy=(1, 1), color='r', size=14)
    vis.show()    
