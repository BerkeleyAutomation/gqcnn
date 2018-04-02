import os
import logging
import math
import time

import cv2
import numpy as np
import skimage.transform as skt

import tensorflow as tf

IM_FILE = '/nfs/diskstation/vsatish/dex-net/data/datasets/yumi/case_00/phoxi/depth_ims_tf_table_00000.npz'
PI = math.pi
NUM_BINS = 16
RESCALE_FACT = 0.25
NUM_TEST_ITERATIONS = 10
GPU_WARM_ITERATIONS = 10

# restart and setup logger
logging.shutdown()
reload(logging)
logging.getLogger().setLevel(logging.INFO)

# load single image for benchmark
logging.info('Loading Image, Thresholding, and Rescaling')
raw_im = np.load(IM_FILE)['arr_0'][0:1]
thresh_im = np.copy(raw_im)
thresh_im[np.where(thresh_im > 1.0)] = 1.0
rescale_image = skt.rescale(thresh_im[0, ..., 0], RESCALE_FACT)
rescale_im = np.zeros((1,) + rescale_image.shape + (1,))
rescale_im[0] = rescale_im

# build rotation matrices for CPU test
logging.info('Building Rotation Matrices for CPU Benchmark')
bin_width = PI / NUM_BINS
rot_mats = None
for i in range(NUM_BINS):
    rot_mat = cv2.getRotationMatrix2D(((rescale_im.shape[2] - 1) / 2, (rescale_im.shape[1] - 1) / 2), (i * bin_width + bin_width / 2) * 180 / PI, 1)
    if rot_mats is None:
        rot_mats = np.zeros((NUM_BINS,) + rot_mat.shape)
    rot_mats[i] = rot_mat

# benchmark rotation on CPU
logging.info('Beginning CPU Benchmark')
cpu_times = np.zeros((NUM_TEST_ITERATIONS,))
for i in range(NUM_TEST_ITERATIONS):
    start_time = time.time()
    for j in range(NUM_BINS):
        rot_im = cv2.warpAffine(rescale_im[0, ..., 0], rot_mats[j], (rescale_im.shape[2], rescale_im.shape[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cpu_times[i] = time.time() - start_time
    logging.info('CPU Iteration {} took {} seconds'.format(i, cpu_times[i]))
logging.info('Average CPU time was {} seconds'.format(np.mean(cpu_times)))

# build Tensorflow graph for GPU test
logging.info('Building Tf Graph for GPU Test')
input_ims = tf.placeholder(tf.float32, shape=[NUM_BINS, rescale_im.shape[1], rescale_im.shape[2], rescale_im.shape[3]])
input_angles = tf.placeholder(tf.float32, shape=[NUM_BINS])
rot_ims = tf.contrib.image.rotate(input_ims, input_angles)

# caculate rotation angles
logging.info('Calculating rotation angles')
rot_angles = np.zeros((NUM_BINS,))
for i in range(NUM_BINS):
    rot_angles[i] = (i * bin_width + bin_width / 2)

# open TF session
logging.info('Creating TF session')
sess = tf.Session()

# warm-up GPU
logging.info('Warming-up GPU')
for i in range(GPU_WARM_ITERATIONS):
    start_time = time.time()
    sess.run(rot_ims, feed_dict={input_ims: np.tile(rescale_im, [NUM_BINS, 1, 1, 1]), input_angles: rot_angles})
    logging.info('Warm-up iteration {} took {} seconds'.format(i, time.time() - start_time))

# benchmark rotation on GPU
logging.info('Beginning GPU Benchmark')
gpu_times = np.zeros((NUM_TEST_ITERATIONS,))
for i in range(NUM_TEST_ITERATIONS):
    start_time = time.time()
    sess.run(rot_ims, feed_dict={input_ims: np.tile(rescale_im, [NUM_BINS, 1, 1, 1]), input_angles: rot_angles})
    gpu_times[i] = time.time() - start_time
    logging.info('GPU Iteration {} took {} seconds'.format(i, gpu_times[i]))
logging.info('Average GPU time was {} seconds'.format(np.mean(gpu_times))) 

# close TF session
logging.info('Closing TF Session')
sess.close()

logging.info('Test Finished!')
