"""
Script for converting a tensor dataset of bin images to a legacy GQCNN dataset for training
fully-connected GQCNN networks.
Author: Vishal Satish
"""
import logging
import numpy as np
import os
import time
import IPython

from dexnet.learning import TensorDataset
from perception import DepthImage

IM_DATASET_PATH = '/nfs/diskstation/projects/dex-net/parallel_jaws/datasets/dexnet_4/phoxi_v13/images'
GRASP_DATASET_PATH = '/nfs/diskstation/projects/dex-net/parallel_jaws/datasets/dexnet_4/phoxi_v13/grasps'
CROP_SIZE = 46
NUM_CHANNELS = 1
IM_PER_FILE = 1000
POSE_DIM = 6
GQCNN_SCALE_FACTOR = 0.5
FNAME_PLACE = 5
IM_FILE_TEMPLATE = 'depth_ims_tf_table'
POSE_FILE_TEMPLATE = 'hand_poses'
METRIC_FILE_TEMPLATE = 'robust_wrench_resistance'
OUTPUT_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/phoxi_v13_46x46_03_01_18'
ROTATE = 1

if __name__ == '__main__':
    # setup logger
    logging.getLogger().setLevel(logging.INFO)    

    # get start time
    gen_start_time = time.time()

    # open datasets
    logging.info('Opening datasets')
    im_dataset = TensorDataset.open(IM_DATASET_PATH)
    grasp_dataset = TensorDataset.open(GRASP_DATASET_PATH)
    
    # generate buffers
    logging.info('Allocating buffers')
    im_buffer = np.zeros((IM_PER_FILE, CROP_SIZE, CROP_SIZE, NUM_CHANNELS))
    pose_buffer = np.zeros((IM_PER_FILE, POSE_DIM))
    metric_buffer = np.zeros((IM_PER_FILE,))

    # iterate through the image dataset
    buffer_ind = 0
    out_file_idx = 0
    for im_idx, datum in enumerate(im_dataset):
        im = DepthImage(datum['depth_ims'])
        grasp_start_ind = datum['grasp_start_ind']
        grasp_end_ind = datum['grasp_end_ind']
        
        # iterate through the corresponding grasps
        for i in range(grasp_start_ind, grasp_end_ind + 1):
            grasp = grasp_dataset[i]
            metric = grasp['grasp_metrics']
            pose = grasp['grasps']            

            # align grasp
            logging.info('Aligning grasp {} of {} for image {} of {}'.format(i - grasp_start_ind, grasp_end_ind - grasp_start_ind, im_idx, im_dataset.num_datapoints))
            
            if ROTATE:
                rot_ang = pose[3]
            else:
                rot_ang = 0
            tf_im = im.align(GQCNN_SCALE_FACTOR, np.asarray([pose[1], pose[0]]), rot_ang, CROP_SIZE, CROP_SIZE)

            # add to buffers
            im_buffer[buffer_ind, ...] = tf_im.raw_data
            pose_buffer[buffer_ind, ...] = pose
            metric_buffer[buffer_ind, ...] = metric
            buffer_ind += 1            

            # write out when buffers are full
            if buffer_ind == IM_PER_FILE:
                # dump
                logging.info('Saving {} datapoints'.format(IM_PER_FILE))
                im_fname = '{}_{}'.format(IM_FILE_TEMPLATE, str(out_file_idx).zfill(FNAME_PLACE))
                pose_fname = '{}_{}'.format(POSE_FILE_TEMPLATE, str(out_file_idx).zfill(FNAME_PLACE))
                metric_fname = '{}_{}'.format(METRIC_FILE_TEMPLATE, str(out_file_idx).zfill(FNAME_PLACE))
                np.savez_compressed(os.path.join(OUTPUT_DIR, im_fname), im_buffer)
                np.savez_compressed(os.path.join(OUTPUT_DIR, pose_fname), pose_buffer)
                np.savez_compressed(os.path.join(OUTPUT_DIR, metric_fname), metric_buffer)
                out_file_idx += 1
                buffer_ind = 0
    logging.info('Dataset generation took {} seconds'.format(time.time() - gen_start_time))
