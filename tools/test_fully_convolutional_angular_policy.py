'''
Script to test FullyConvolutionalAngularPolicy.
Author: Vishal Satish
'''
import os
import logging
import time

import skimage.transform as skt
import numpy as np

from autolab_core import YamlConfig
from gqcnn import FullyConvolutionalAngularPolicy, RgbdImageState
from perception import CameraIntrinsics, RgbdImage 

# get test start time
test_start_time = time.time()

# define any enums and constants
TEST_CONFIG = '/home/vsatish/Workspace/dev/gqcnn/cfg/tools/test_fully_convolutional_angular_policy.yaml'
IM_FILE_TEMPLATE = 'depth_ims_tf_table'

# restart and setup logger
logging.shutdown()
reload(logging)
logging.getLogger().setLevel(logging.INFO)

# load and parse config
logging.info('Loading and parsing config...')
test_cfg = YamlConfig(TEST_CONFIG)
dataset_dir = test_cfg['dataset_dir']
num_test_samples = test_cfg['num_test_samples']
num_warmup_iterations = test_cfg['num_warmup_iterations']
num_test_iterations = test_cfg['num_test_iterations']
max_depth_thresh = test_cfg['max_depth_thresh']
rescale_fact = test_cfg['rescale_fact']
camera_intr_file = test_cfg['camera_intr']
camera_intr_rescale_fact = test_cfg['camera_intr_rescale_fact']
policy_cfg = test_cfg['policy_config']

############################################# SAMPLE IMAGES #############################################
# get all filenames
logging.info('Reading all filenames...')
all_filenames = os.listdir(dataset_dir)
im_filenames = [f for f in all_filenames if f.find(IM_FILE_TEMPLATE) > -1]
im_filenames.sort(key=lambda x: int(x[-9:-4]))

# sample desired number of images
logging.info('Sampling test images...')
images = None
counter = 0
while counter < num_test_samples:
    file_num = np.random.randint(len(im_filenames))
    im_data = np.load(os.path.join(dataset_dir, im_filenames[file_num]))['arr_0']
    index = np.random.randint(im_data.shape[0])
 
    # threshold and re-scale the image
    im = im_data[index, ..., 0]
    im[np.where(im > max_depth_thresh)] = max_depth_thresh
    im = skt.rescale(im, rescale_fact)
 
    if images is None:
        images = np.zeros((num_test_samples,) + im.shape + im_data.shape[3:])
    images[counter, ..., 0] = im

    counter += 1

# wrap images into RgbdImageStates
logging.info('Generating wrapped RgbdImageStates...')
camera_intr = CameraIntrinsics.load(camera_intr_file).resize(camera_intr_rescale_fact)
im_states = []
rgbd_raw_data_template = np.zeros(images[0].shape[:-1] + (4,))
for i in range(num_test_samples):
    rgbd_raw_data_template[:, :, 3] = images[i, :, :, 0]
    im_states.append(RgbdImageState(RgbdImage(rgbd_raw_data_template, frame='phoxi'), camera_intr))

############################################# ROLL OUT POLICY #############################################
# create policy
logging.info('Creating policy...')
policy = FullyConvolutionalAngularPolicy(policy_cfg)

# warm up GPUs
logging.info('Warming-up GPUs...')
for i in range(num_warmup_iterations):
    start_time = time.time()
    for j in range(num_test_samples):
        policy(im_states[j])
    logging.info('Policy warm-up iteration {} took {} seconds'.format(i, time.time() - start_time))

# roll out policy
policy_times = np.zeros((num_test_iterations,))
actions = []
logging.info('Rolling out policy...')
for i in range(num_test_iterations):
    start_time = time.time()
    for j in range(num_test_samples):
        actions.append(policy(im_states[j]))
    policy_times[i] = time.time() - start_time
    logging.info('Policy iteration {} took {} seconds'.format(i, policy_times[i]))
    actions = []
logging.info('Average policy iteration time was {} seconds'.format(np.mean(policy_times)))

# clean up
logging.info('Cleaning up policy...')
# delete policy to close internal GQ-CNN Tensorflow session
del policy
logging.info('Total test time was {} seconds.'.format(time.time() - test_start_time))
