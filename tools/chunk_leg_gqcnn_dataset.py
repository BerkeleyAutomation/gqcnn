"""
Script for modifying stored file size of legacy GQCNN dataset.
Author: Vishal Satish
"""
import logging
import numpy as np
import os
import time
import IPython

import matplotlib.pyplot as plt

DATASET_PATH = '/nfs/diskstation/vsatish/dex-net/data/datasets/dexnet_2_fc_no_rot_02_13_18_mini/'
IM_PER_FILE = 100
FNAME_PLACE = 6
IM_FILE_TEMPLATE = 'depth_ims_tf_table'
POSE_FILE_TEMPLATE = 'hand_poses'
METRIC_FILE_TEMPLATE = 'robust_ferrari_canny'
OUTPUT_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/dexnet_2_fc_no_rot_mini_random_depths_pos_04_29_18/'
#OUTPUT_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/dexnet_2_fc_no_rot_mini_04_29_18/'
#OUTPUT_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/test_dump/'
NUM_RANDOM_DEPTHS = 5
GRIPPER_DEPTH = 0.07
METRIC_THRESH = 0.1
VIS = 0
DEBUG = 0

if __name__ == '__main__':
    # setup logger
    logging.getLogger().setLevel(logging.INFO)    

    # get start time
    gen_start_time = time.time()

    # read filenames
    logging.info('Reading filenames')
    all_filenames = os.listdir(DATASET_PATH)
    im_filenames = [f for f in all_filenames if f.find(IM_FILE_TEMPLATE) > -1]
    pose_filenames = [f for f in all_filenames if f.find(POSE_FILE_TEMPLATE) > -1]
    metric_filenames = [f for f in all_filenames if f.find(METRIC_FILE_TEMPLATE) > -1]
    im_filenames.sort(key=lambda x: int(x[-9:-4]))
    pose_filenames.sort(key=lambda x: int(x[-9:-4]))
    metric_filenames.sort(key=lambda x: int(x[-9:-4]))
    
    # init buffers
    logging.info('Initializing buffers')
    im_buffer = None
    pose_buffer = None
    metric_buffer = None

    # iterate through the dataset file-by-file and save with the new IM_PER_FILE
    buffer_ind = 0
    out_file_idx = 0
    for im_fname, pose_fname, metric_fname in zip(im_filenames, pose_filenames, metric_filenames):
        # load data
        im_data = np.load(os.path.join(DATASET_PATH, im_fname))['arr_0']
        pose_data = np.load(os.path.join(DATASET_PATH, pose_fname))['arr_0']
        metric_data = np.load(os.path.join(DATASET_PATH, metric_fname))['arr_0']
        
        num_processed = 0
        num_total = im_data.shape[0]
        data_ind = 0
        while num_processed < num_total:
            # get number of datapoints to load into buffers
            num_to_load = min(max(0, IM_PER_FILE - buffer_ind), num_total - data_ind)
            
            # allocate buffers if needed
            if im_buffer is None:
                logging.info('Allocating buffers')
                im_buffer = np.zeros((IM_PER_FILE + IM_PER_FILE * NUM_RANDOM_DEPTHS,) + im_data.shape[1:])
                pose_buffer = np.zeros((IM_PER_FILE + IM_PER_FILE * NUM_RANDOM_DEPTHS,) + pose_data.shape[1:])
                metric_buffer = np.zeros((IM_PER_FILE + IM_PER_FILE * NUM_RANDOM_DEPTHS,))

            # vis datapoints
            if VIS:
                plt.figure()
                for i in range(num_to_load):
                    logging.info('Visualizing image {} of {} for file {}'.format(data_ind + i + 1, num_total, im_fname))
                    logging.info('Pose: {}'.format(pose_data[data_ind + i]))
                    logging.info('Metric: {}'.format(metric_data[data_ind + i]))
                    plt.clf()
                    plt.imshow(im_data[data_ind + i, ..., 0], cmap=plt.cm.gray)
                    plt.show()
 
            # generate random grasps depth-wise
            if NUM_RANDOM_DEPTHS > 0:
                logging.info('Generating random depths...')
                for j in range(num_to_load):
                    if metric_data[data_ind + j] > METRIC_THRESH:
                        im = im_data[data_ind + j]
                        exc_range_low = im[im.shape[0] // 2, im.shape[1] // 2, 0]
                        exc_range_high = exc_range_low + GRIPPER_DEPTH
                        pose = np.copy(pose_data[data_ind + j])
                        for i in range(NUM_RANDOM_DEPTHS):
                            logging.info('Generating random depth {} of {}'.format(i + 1, NUM_RANDOM_DEPTHS))
                            rand_depth = np.random.random()
                            while exc_range_low < rand_depth < exc_range_high:
                                rand_depth = np.random.random()
                            if DEBUG:
                                logging.info('Sampled random depth: {}'.format(rand_depth))

                            # generate new pose with random depth
                            pose[2] = rand_depth                    

                            # add to buffers
                            im_buffer[buffer_ind] = im
                            pose_buffer[buffer_ind] = pose
                            metric_buffer[buffer_ind] = 0.0
                            buffer_ind += 1
 
            # fill buffers
            im_buffer[buffer_ind:buffer_ind + num_to_load] = im_data[data_ind:data_ind + num_to_load]
            pose_buffer[buffer_ind:buffer_ind + num_to_load] = pose_data[data_ind:data_ind + num_to_load]
            metric_buffer[buffer_ind:buffer_ind + num_to_load] = metric_data[data_ind:data_ind + num_to_load]
            
            # update indices
            buffer_ind += num_to_load
            data_ind += num_to_load
            num_processed += num_to_load            
            
            # write out when buffers are full
            if buffer_ind >= IM_PER_FILE:
                # dump IM_PER_FILE datums
                logging.info('Saving {} datapoints for file {}'.format(IM_PER_FILE, im_fname))
                im_filename = '{}_{}'.format(IM_FILE_TEMPLATE, str(out_file_idx).zfill(FNAME_PLACE))
                pose_filename = '{}_{}'.format(POSE_FILE_TEMPLATE, str(out_file_idx).zfill(FNAME_PLACE))
                metric_filename = '{}_{}'.format(METRIC_FILE_TEMPLATE, str(out_file_idx).zfill(FNAME_PLACE))
                np.savez_compressed(os.path.join(OUTPUT_DIR, im_filename), im_buffer[:IM_PER_FILE])
                np.savez_compressed(os.path.join(OUTPUT_DIR, pose_filename), pose_buffer[:IM_PER_FILE])
                np.savez_compressed(os.path.join(OUTPUT_DIR, metric_filename), metric_buffer[:IM_PER_FILE])
                out_file_idx += 1
                im_buffer[:buffer_ind - IM_PER_FILE] = im_buffer[IM_PER_FILE:buffer_ind]
                pose_buffer[:buffer_ind - IM_PER_FILE] = pose_buffer[IM_PER_FILE:buffer_ind]
                metric_buffer[:buffer_ind - IM_PER_FILE] = metric_buffer[IM_PER_FILE:buffer_ind]
                buffer_ind = buffer_ind - IM_PER_FILE
             
    logging.info('Dataset generation took {} seconds'.format(time.time() - gen_start_time))
