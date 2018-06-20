import logging
import os
import math

import numpy as np
import matplotlib.pyplot as plt

from autolab_core import Point
from perception import CameraIntrinsics, DepthImage
from gqcnn.model import get_gqcnn_model
from gqcnn import Visualizer as vis, Grasp2D

MODEL_DIR = '/home/vsatish/Data/dexnet/data/models/test_dump/model_ikwvhwbuhi/'
# MODEL_DIR = '/home/vsatish/Data/dexnet/data/models/test_dump/model_cfxloaqbuj/' # good
# DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/fizzytablets_benchmark_random_fc_ang_iter_3_leg_no_rot_05_21_18/'
DATASET_DIR = '/home/vsatish/Workspace/dev/gqcnn/test_dump/'
CAMERA_INTR_DIR =  '/nfs/diskstation/calib/phoxi/phoxi.intr'
IM_FILE_TEMPLATE = 'depth_ims_tf_table'
POSE_FILE_TEMPLATE = 'hand_poses'
METRIC_FILE_TEMPLATE = 'robust_wrench_resistance'
NUM_TEST_SAMPLES = 10
POSE_DIM = 1
NUM_BINS = 16
ONLY_POSITIVES = 0
METRIC_THRESH = 0.75
pose_parser = lambda p: p[2:3]
GRIPPER_WIDTH = 0.05
DEBUG = 0
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
    if ONLY_POSITIVES and metric < METRIC_THRESH:
        continue
    images[counter, ...] = im
    poses[counter, ...] = pose_parser(pose)
    ground_truth_angles[counter] = pose[3]
    metrics[counter] = metric
    counter += 1

# poses = np.tile(np.asarray([[0.2]]), (NUM_TEST_SAMPLES, 1))

# load Angular-GQCNN model
logging.info('Loading Angular-GQCNN')
ang_gqcnn = get_gqcnn_model().load(MODEL_DIR)

# pair-wise softmax function
def pairwise_softmax(tmp):
    softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
    out = None
    for i in range(0, tmp.shape[0], 2):
        if out is None:
            out = softmax(tmp[i:i + 2])
        else:
            out = np.r_[out, softmax(tmp[i:i + 2])]     
    return out

# predict test images
logging.info('Inferring')
ang_gqcnn.open_session()
angular_pred = ang_gqcnn.predict(images, poses)
ang_gqcnn.close_session()

# apply pair-wise softmax
soft_pred = np.zeros_like(angular_pred)
for i in range(angular_pred.shape[0]):
    soft_pred[i] = pairwise_softmax(angular_pred[i])

# generate grasps
logging.info('Generating grasp visualizations')
grasps = []
bin_width = math.pi / NUM_BINS
center = Point(np.asarray([images.shape[1] / 2, images.shape[2] / 2]))
camera_intr = CameraIntrinsics.load(CAMERA_INTR_DIR).resize(1.0)
for i in range(NUM_TEST_SAMPLES):
    sample_grasps = []
    for j in range(NUM_BINS):
        bin_cent_ang = j * bin_width + bin_width / 2
#        logging.info('Bin width: {}, Ang: {}, Shift ang: {}'.format(bin_width, bin_cent_ang, bin_cent_ang - math.pi / 4))
        sample_grasps.append(Grasp2D(center, math.pi / 2 - bin_cent_ang, poses[i, 0], width=GRIPPER_WIDTH, camera_intr=camera_intr))
    sample_grasps.append(Grasp2D(center, ground_truth_angles[i], poses[i, 0], width=GRIPPER_WIDTH, camera_intr=camera_intr))
    grasps.append(sample_grasps)

# visualize
logging.info('Beginning Visualization')
vis.figure(size=(10, 10))
for i in range(NUM_TEST_SAMPLES):
    logging.info('Angular Pred: {}'.format(angular_pred[i]))
    logging.info('Depth: {}'.format(poses[i, 0]))
    vis.clf()
    vis.imshow(DepthImage(images[i]))
    for j, grasp in enumerate(grasps[i][:-1]):
        vis.grasp(grasp, scale=0.5, show_axis=True, color=plt.cm.RdYlGn(angular_pred[i, j * 2 + 1]))
    vis.grasp(grasps[i][-1], scale=0.5, show_axis=True, color='black')
    plt.annotate(str(metrics[i]), xy=(1, 1), color='r', size=14)
    vis.show()    
