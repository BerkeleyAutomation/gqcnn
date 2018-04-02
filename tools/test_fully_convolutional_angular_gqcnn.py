"""
Script to benchmark normal gqcnn, fully-convolutional gqcnn, and fully-convolutional angular gqcnn
Author: Vishal Satish
"""
import os
import logging
import math
import time
import sys

import skimage.transform as skt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pplot
import cv2 as cv

from autolab_core import Point
from perception import CameraIntrinsics, DepthImage
from gqcnn import Visualizer as vis, Grasp2D
from gqcnn.model import get_gqcnn_model

# restart and setup logger
logging.shutdown()
reload(logging)
logging.getLogger().setLevel(logging.INFO)

# config params
DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/yumi/case_00/phoxi'
NORMAL_MODEL_DIR =  '/home/vsatish/Data/dexnet/data/models/test_dump/model_ebhsmdqmjd'
ANGULAR_MODEL_DIR = '/home/vsatish/Data/dexnet/data/models/test_dump/model_tiqeyfvwox'
CAMERA_INTR_DIR =  '/nfs/diskstation/calib/phoxi/phoxi.intr'
CAMERA_INTR_RESCALE_FACT = 0.25
GRIPPER_WIDTH = 0.05
NUM_TEST_SAMPLES = 2
IM_FILE_TEMPLATE = 'depth_ims_tf_table'
POSE_FILE_TEMPLATE = 'hand_poses'
RESCALE_FACT = 0.25
DEPTH_THRESH = 1.0
NUM_CROPS = 107*75
CROP_W = 46
CROP_STRIDE = 2
FULLY_CONV_CONFIG = {'im_width': 258, 'im_height': 193}
VIS = 1
VIS_POSE_LOC = (5, 5)
FIG_SIZE = (15, 15)
# INFERENCE_DEPTHS = np.asarray([[0.8], [0.8], [0.8], [0.8], [0.8], [0.8], [0.8], [0.8], [0.8], [0.8], [0.8], [0.8], [0.8], [0.8], [0.8], [0.8]])
# INFERENCE_DEPTHS = np.asarray([[0.8]])
INFERENCE_DEPTHS = np.asarray([[0.8], [0.8]])
DEPTH_PARSER = lambda p: p[2:3]
DEPTH_PARSER_MULTI_DIM = lambda p: p[:, 2:3]
ANGLE_PARSER = lambda p: p[:, 3]
NUM_ANGULAR_BINS = 16
PI = math.pi
PI_2 = math.pi / 2
NUM_TEST_ITERATIONS = 1
GPU_WARM_ITERATIONS = 1
GRASP_SUCCESS_THRESH = 0.3

ROT_IMAGES = 1
TEST_NORMAL_GQCNN = 1
TEST_FULLY_CONV_GQCNN = 1
TEST_FULLY_CONV_ANG_GQCNN = 1

# get test start time
test_start_time = time.time()

############################################# SAMPLE IMAGES #############################################
# get all filenames
logging.info('Reading all filenames')
all_filenames = os.listdir(DATASET_DIR)
im_filenames = [f for f in all_filenames if f.find(IM_FILE_TEMPLATE) > -1]
pose_filenames = [f for f in all_filenames if f.find(POSE_FILE_TEMPLATE) > -1]
im_filenames.sort(key=lambda x: int(x[-9:-4]))
pose_filenames.sort(key=lambda x: int(x[-9:-4]))
 
# sample NUM_TEST_SAMPLES images
logging.info('Sampling test images')
images = None
poses = None
counter = 0
while counter < NUM_TEST_SAMPLES:
    file_num = np.random.randint(len(im_filenames))
    im_data = np.load(os.path.join(DATASET_DIR, im_filenames[file_num]))['arr_0']
    pose_data = np.load(os.path.join(DATASET_DIR, pose_filenames[file_num]))['arr_0']
    index = np.random.randint(im_data.shape[0])

    # threshold and re-scale the image
    im = im_data[index, ..., 0]
    im[np.where(im > DEPTH_THRESH)] = DEPTH_THRESH
    im = skt.rescale(im, RESCALE_FACT)

    if images is None:
        images = np.zeros((NUM_TEST_SAMPLES,) + im.shape + im_data.shape[3:])
        poses = np.zeros((NUM_TEST_SAMPLES,) + pose_data.shape[1:])
    images[counter, ..., 0] = im
    poses[counter, ...] = pose_data[index]
    counter += 1

# make sure there are the same number of supplied INFERENCE DEPTHS as sampled IMAGES
assert INFERENCE_DEPTHS.shape[0] == images.shape[0], 'Number of supplied inference depths must match number of test samples'


############################################# ROTATE IMAGES #############################################
if ROT_IMAGES:
    logging.info('Setting up for GPU Image Rotation')

    # build Tensorflow graph for GPU test
    logging.info('Building TF Graph')
    input_ims = tf.placeholder(tf.float32, shape=[NUM_ANGULAR_BINS, images.shape[1], images.shape[2], images.shape[3]])
    input_angles = tf.placeholder(tf.float32, shape=[NUM_ANGULAR_BINS])
    rot_ims = tf.contrib.image.rotate(input_ims, input_angles)

    # calculate rotation angles
    logging.info('Calculating rotation angles')
    rot_angles = np.zeros((NUM_ANGULAR_BINS,))
    bin_width = PI / NUM_ANGULAR_BINS
    for i in range(NUM_ANGULAR_BINS):
        rot_angles[i] = PI_2 - (i * bin_width + bin_width / 2)

    # open TF session
    logging.info('Creating TF session')
    sess = tf.Session()

    # warm-up GPU
    logging.info('Warming-up GPU')
    for i in range(GPU_WARM_ITERATIONS):
        start_time = time.time()
        for j in range(NUM_TEST_SAMPLES):
            sess.run(rot_ims, feed_dict={input_ims: np.tile(images[j:j+1], [NUM_ANGULAR_BINS, 1, 1, 1]), input_angles: rot_angles})
        logging.info('Rotation warm-up iteration {} took {} seconds'.format(i, time.time() - start_time))

    # benchmark rotation on GPU
    logging.info('Beginning GPU Rotation')
    rot_times = np.zeros((NUM_TEST_ITERATIONS,))
    rot_images = np.zeros((NUM_TEST_SAMPLES, NUM_ANGULAR_BINS) + images.shape[1:])
    for i in range(NUM_TEST_ITERATIONS):
        start_time = time.time()
        for j in range(NUM_TEST_SAMPLES):
            rot_images[j] = sess.run(rot_ims, feed_dict={input_ims: np.tile(images[j:j+1], [NUM_ANGULAR_BINS, 1, 1, 1]), input_angles: rot_angles})
        rot_times[i] = time.time() - start_time
        logging.info('Rotation Iteration {} took {} seconds'.format(i, rot_times[i]))
    avg_rot_time = np.mean(rot_times)
    logging.info('Average rotation time was {} seconds'.format(np.mean(avg_rot_time)))

    # close TF session
    logging.info('Closing TF Session')
    sess.close() 
else:
    logging.info('Skipping Image Rotation')
    logging.warning('ONLY FULLY-CONVOLUTIONAL ANGULAR GQCNN CAN BE TESTED WITHOUT IMAGE ROTATION')

############################################# BENCHMARK NORMAL GQCNN #############################################
if TEST_NORMAL_GQCNN and ROT_IMAGES:
    logging.info('Testing Normal GQCNN')

    # first reshape rotated images and poses
    normal_gqcnn_images = rot_images.reshape((NUM_TEST_SAMPLES * NUM_ANGULAR_BINS,) + rot_images.shape[2:])
    normal_gqcnn_poses = np.repeat(INFERENCE_DEPTHS, NUM_ANGULAR_BINS, axis=0)

    # crop images
    logging.info('Cropping Images')
    crop_images = np.zeros((normal_gqcnn_images.shape[0]*NUM_CROPS, CROP_W, CROP_W, normal_gqcnn_images.shape[3]))
    crop_poses = np.zeros((normal_gqcnn_images.shape[0]*NUM_CROPS, 1))
    centers = np.zeros((NUM_CROPS, 2))
    calc_centers = True
    crop_times = np.zeros((NUM_TEST_ITERATIONS,))
    for j in range(NUM_TEST_ITERATIONS):
        start_time = time.time()
        for i in range(normal_gqcnn_images.shape[0]):
            im = normal_gqcnn_images[i]
            width = normal_gqcnn_images.shape[2]
            height = normal_gqcnn_images.shape[1]
            index = 0
            for h in range(0, height - CROP_W + CROP_STRIDE, CROP_STRIDE):
                for w in range(0, width - CROP_W + CROP_STRIDE, CROP_STRIDE):
                    crop = im[h:h + CROP_W, w:w + CROP_W, ...]
                    if (crop.shape[0] != CROP_W):
                        crop_images[i * NUM_CROPS + index] = np.ones((CROP_W, CROP_W, normal_gqcnn_images.shape[-1]))
                    else:
                        crop_images[i * NUM_CROPS + index] = im[h:h + CROP_W, w:w + CROP_W, ...]
                    if calc_centers:
                        centers[index] = np.asarray([h + CROP_W / 2, w + CROP_W / 2])
                    index += 1
            calc_centers = False
            crop_poses[i * NUM_CROPS:i * NUM_CROPS + NUM_CROPS] = normal_gqcnn_poses[i]
        crop_times[j] = time.time() - start_time
        logging.info('Cropping Iteration {} took {} seconds'.format(j, crop_times[j]))
    avg_crop_time = np.mean(crop_times)
    logging.info('Average crop time was {} seconds'.format(avg_crop_time))
    
    # create GQCNN
    logging.info('Creating Normal GQCNN')
    normal_gqcnn = get_gqcnn_model().load(NORMAL_MODEL_DIR)
    
    # open GQCNN TF session
    logging.info('Creating TF Session')
    normal_gqcnn.open_session()
    
    # warm-up GPU for inference
    logging.info('Warming Up GPU')
    for i in range(GPU_WARM_ITERATIONS):
        start_time = time.time()
        normal_gqcnn.predict(crop_images, crop_poses)
        logging.info('Normal GQCNN Inference warm-up iteration {} took {} seconds'.format(i, time.time() - start_time))
    
    # infer
    logging.info('Inferring')
    normal_inference_times = np.zeros((NUM_TEST_ITERATIONS,)) 
    for i in range(NUM_TEST_ITERATIONS): 
        start_time = time.time() 
        normal_gqcnn_pred = normal_gqcnn.predict(crop_images, crop_poses)
        normal_inference_times[i] = time.time() - start_time
        logging.info('Normal GQCNN Inference iteration {} took {} seconds'.format(i, normal_inference_times[i]))
    avg_normal_inference_time = np.mean(normal_inference_times)
    logging.info('Average Normal GQCNN Inference time was {} seconds'.format(avg_normal_inference_time))
    
    # close GQCNN TF session
    logging.info('Closing TF Session')
    normal_gqcnn.close_session()
elif TEST_NORMAL_GQCNN:
    logging.error('NORMAL GQCNN CANNOT BE TESTED WITHOUT ROTATING IMAGES')
    sys.exit(0)
else:
    logging.info('Skipping normal GQCNN test')

############################################# BENCHMARK FULLY-CONVOLUTIONAL GQCNN #############################################
if TEST_FULLY_CONV_GQCNN and ROT_IMAGES:
    logging.info('Testing Fully-Convolutional GQCNN')

    # first reshape rotated images
    fully_conv_gqcnn_images = rot_images.reshape((NUM_TEST_SAMPLES * NUM_ANGULAR_BINS,) + rot_images.shape[2:])
    fully_conv_gqcnn_poses = np.repeat(INFERENCE_DEPTHS, NUM_ANGULAR_BINS, axis=0)
    
    # create GQCNN
    logging.info('Creating Fully-Convolutional GQCNN')
    fully_conv_gqcnn = get_gqcnn_model().load(NORMAL_MODEL_DIR, fully_conv_config=FULLY_CONV_CONFIG)
    
    # open GQCNN TF session
    logging.info('Creating TF Session')
    fully_conv_gqcnn.open_session()
    
    # warm-up GPU for inference
    logging.info('Warming Up GPU')
    for i in range(GPU_WARM_ITERATIONS):
        start_time = time.time()
        fully_conv_gqcnn.predict(fully_conv_gqcnn_images, fully_conv_gqcnn_poses)
        logging.info('Fully-Convolutional GQCNN Inference warm-up iteration {} took {} seconds'.format(i, time.time() - start_time))
    
    # infer
    logging.info('Inferring')
    fully_conv_inference_times = np.zeros((NUM_TEST_ITERATIONS,))
    for i in range(NUM_TEST_ITERATIONS):
        start_time = time.time()
        fully_conv_gqcnn_pred = fully_conv_gqcnn.predict(fully_conv_gqcnn_images, fully_conv_gqcnn_poses)
        fully_conv_inference_times[i] = time.time() - start_time
        logging.info('Fully-Convolutional GQCNN Inference iteration {} took {} seconds'.format(i, fully_conv_inference_times[i]))
    avg_fully_conv_inference_time = np.mean(fully_conv_inference_times)
    logging.info('Average Fully-Convolutional GQCNN Inference time was {} seconds'.format(avg_fully_conv_inference_time))
    
    # close GQCNN TF session
    logging.info('Closing TF Session')
    fully_conv_gqcnn.close_session()
elif TEST_FULLY_CONV_GQCNN:
    logging.error('FULLY CONVOLUTIONAL GQCNN CANNOT BE TESTED WITHOUT ROTATING IMAGES')
    sys.exit(0)
else:
    logging.info('Skipping fully-convolutional GQCNN test')
 
############################################# BENCHMARK FULLY-CONVOLUTIONAL-ANGULAR GQCNN #############################################
if TEST_FULLY_CONV_ANG_GQCNN:
    logging.info('Testing Fully-Convolutional Angular GQCNN')
    
    fully_conv_ang_gqcnn_images = images
    fully_conv_ang_gqcnn_poses = INFERENCE_DEPTHS
    
    # create GQCNN
    logging.info('Creating Fully-Convolutional Angular GQCNN')
    fully_conv_ang_gqcnn = get_gqcnn_model().load(ANGULAR_MODEL_DIR, fully_conv_config=FULLY_CONV_CONFIG)
    
    # open GQCNN TF session
    logging.info('Creating TF Session')
    fully_conv_ang_gqcnn.open_session()
    
    # warm-up GPU for inference
    logging.info('Warming Up GPU')
    for i in range(GPU_WARM_ITERATIONS):
        start_time = time.time()
        fully_conv_ang_gqcnn.predict(fully_conv_ang_gqcnn_images, fully_conv_ang_gqcnn_poses)
        logging.info('Fully-Convolutional Angular GQCNN Inference warm-up iteration {} took {} seconds'.format(i, time.time() - start_time))
    
    # infer
    logging.info('Inferring')
    fully_conv_ang_inference_times = np.zeros((NUM_TEST_ITERATIONS,))
    for i in range(NUM_TEST_ITERATIONS):
        start_time = time.time()
        fully_conv_ang_pred = fully_conv_ang_gqcnn.predict(fully_conv_ang_gqcnn_images, fully_conv_ang_gqcnn_poses)
        fully_conv_ang_inference_times[i] = time.time() - start_time
        logging.info('Fully-Convolutional Angular GQCNN Inference iteration {} took {} seconds'.format(i, fully_conv_ang_inference_times[i]))
    avg_fully_conv_ang_inference_time = np.mean(fully_conv_ang_inference_times)
    logging.info('Average Fully-Convolutional Angular GQCNN Inference Time was {} seconds'.format(avg_fully_conv_ang_inference_time))
    
    # close GQCNN TF session
    logging.info('Closing TF Session')
    fully_conv_ang_gqcnn.close_session()
else:
    logging.info('Skipping fully-convolutional angular GQCNN test')

def plot_grasps(im, preds, grasps, plt, plt_title, rot_im=False, rot_mats=None):
    logging.info('Plotting grasps for {}'.format(plt_title))
    plt.imshow(DepthImage(im))
    for j in range(preds.shape[1]):
        for k in range(preds.shape[2]):
            for l in range(NUM_ANGULAR_BINS):
                if rot_im:
                    rot_mat = rot_mats[l]
                    n_j = int(rot_mat[1, 0] * k + rot_mat[1, 1] * j + rot_mat[1, 2])
                    n_k = int(rot_mat[0, 0] * k + rot_mat[0, 1] * j + rot_mat[0, 2])
                else:
                    n_j, n_k = j, k
                if 0 < n_j < preds.shape[1] and 0 < n_k < preds.shape[2] and preds[l, n_j, n_k, 1] > GRASP_SUCCESS_THRESH:
                    plt.grasp(grasps[j][k][l], scale=0.5, show_axis=True, color=pplot.cm.RdYlGn(preds[l, n_j, n_k, 1]))
    plt.title(plt_title)

if VIS:
    # re-shape predictions into shape (NUM_TEST_SAMPLES, NUM_ANGULAR_BINS, OUT_W, OUT_H, 2)
    logging.info('Re-shaping predictions for visualization')
    NEW_SHAPE = (NUM_TEST_SAMPLES, NUM_ANGULAR_BINS, fully_conv_ang_pred.shape[1], fully_conv_ang_pred.shape[2], 2)
    if TEST_NORMAL_GQCNN:
        normal_gqcnn_pred_reshape = normal_gqcnn_pred.reshape(NEW_SHAPE)
    if TEST_FULLY_CONV_GQCNN:
        fully_conv_gqcnn_pred_reshape = np.zeros(NEW_SHAPE)
        for i in range(NUM_TEST_SAMPLES):
            for j in range(NUM_ANGULAR_BINS):
                fully_conv_gqcnn_pred_reshape[i, j, ...] = fully_conv_gqcnn_pred[i * NUM_ANGULAR_BINS + j, ...]
    if TEST_FULLY_CONV_ANG_GQCNN:
        fully_conv_ang_pred_reshape = np.zeros(NEW_SHAPE)
        for i in range(NUM_TEST_SAMPLES):
            for j in range(NUM_ANGULAR_BINS):
                fully_conv_ang_pred_reshape[i, j, ...] = fully_conv_ang_pred[i, :, :, j*2:(j+1)*2]
    
    # generate rotated grasps for visualization
    logging.info('Generating rotated grasp visualizations')
    camera_intr = CameraIntrinsics.load(CAMERA_INTR_DIR).resize(CAMERA_INTR_RESCALE_FACT)
    grasps = []
    bin_width = PI / NUM_ANGULAR_BINS
    height = images.shape[1]
    width = images.shape[2]
    for i in range(NUM_TEST_SAMPLES):
        sample_grasps = []
        for h in range(0, height - CROP_W + CROP_STRIDE, CROP_STRIDE):
            r_grasps = [] 
            for w in range(0, width - CROP_W + CROP_STRIDE, CROP_STRIDE):
                center = Point(np.asarray([w + CROP_W / 2, h + CROP_W / 2]))
                a_grasps = []
                for j in range(NUM_ANGULAR_BINS):
                    bin_cent_ang = j * bin_width + bin_width / 2
                    a_grasps.append(Grasp2D(center, PI / 2 - bin_cent_ang, INFERENCE_DEPTHS[i, 0], width=GRIPPER_WIDTH, camera_intr=camera_intr))
                r_grasps.append(a_grasps)
            sample_grasps.append(r_grasps)
        grasps.append(sample_grasps)
    
    # generate rotation matrices if testing normal gqcnn or fully-convolutional gqcnn
    if TEST_NORMAL_GQCNN or TEST_FULLY_CONV_GQCNN:
        rot_mats = np.zeros((NUM_ANGULAR_BINS, 2, 3))
        for i in range(NUM_ANGULAR_BINS):
            rot_ang = 180 / PI * (PI_2 - (i * bin_width + bin_width / 2))
            # TODO: remove assumption that TEST_FULLY_CONV_GQCNN == 1
            rot_mat = cv.getRotationMatrix2D((int(fully_conv_gqcnn_pred_reshape.shape[3] / 2), int(fully_conv_gqcnn_pred_reshape.shape[2] / 2)), rot_ang, 1.0)
            rot_mats[i] = rot_mat

    # plot
    num_subplots = TEST_NORMAL_GQCNN + TEST_FULLY_CONV_GQCNN + TEST_FULLY_CONV_ANG_GQCNN
    for i in range(NUM_TEST_SAMPLES):
        logging.info('Visualizing Grasps for Image {}'.format(i))
        plot_idx = 1
        fig = vis.figure((15, 40))
        fig.suptitle('Image: {}'.format(i))
        if TEST_NORMAL_GQCNN:
            vis.subplot(100 + num_subplots * 10 + plot_idx)
            plot_idx += 1
            plot_grasps(images[i], normal_gqcnn_pred_reshape[i], grasps[i], vis, 'Normal GQCNN', rot_im=True, rot_mats=rot_mats)
        if TEST_FULLY_CONV_GQCNN:
            vis.subplot(100 + num_subplots * 10 + plot_idx)
            plot_idx += 1
            plot_grasps(images[i], fully_conv_gqcnn_pred_reshape[i], grasps[i], vis, 'Fully-convolutional GQCNN', rot_im=True, rot_mats=rot_mats)
        if TEST_FULLY_CONV_ANG_GQCNN:
            vis.subplot(100 + num_subplots * 10 + plot_idx)
            plot_grasps(images[i], fully_conv_ang_pred_reshape[i], grasps[i], vis, 'Fully-convolutional Angular GQCNN')
        vis.show()

logging.info('Test finished in {} seconds'.format(time.time() - test_start_time))
