import logging
import time
import json
import os

import numpy as np

from gqcnn.model import get_gqcnn_model

# define any enums
NUM_SAMPLES = 1000
MODEL_DIR = '/home/vsatish/Data/dexnet/data/models/test_dump/model_jrtubybmla'
DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/mini_dexnet_all_trans_01_20_17'
CONFIG_FILENAME = 'config.json'
TEST_ITERATIONS = 10
GPU = True
GPU_WARM_ITERATIONS = 10
IM_FILENAME_TEMPLATE = 'depth_ims_tf_table'
POSE_FILENAME_TEMPLATE = 'hand_poses'
POSE_DIM = 1
SAVE_PRED = 1
SAVE_DIR = '.'
SAVE_FNAME = 'pred_tf'
BACKEND = 'tf'

# define function to process pose data for depth value
pose_preproc = lambda x: x[:, 2:3]

if __name__ == '__main__':
    # setup logger
    logging.getLogger().setLevel(logging.INFO)
    
    # load samples
    logging.info('Setting up for test')
    logging.info('Loading {} samples'.format(NUM_SAMPLES))
    
    all_filenames = os.listdir(DATASET_DIR)
    im_filenames = [f for f in all_filenames if f.find(IM_FILENAME_TEMPLATE) > -1]
    pose_filenames = [f for f in all_filenames if f.find(POSE_FILENAME_TEMPLATE) > -1]
    im_filenames.sort(key=lambda x: int(x[-9:-4]))
    pose_filenames.sort(key=lambda x: int(x[-9:-4]))

    im_shape = np.load(os.path.join(DATASET_DIR, im_filenames[0]))['arr_0'][0].shape
    im_tensor = np.zeros((NUM_SAMPLES,) + im_shape)
    pose_tensor = np.zeros((NUM_SAMPLES, POSE_DIM))

    start_ind = 0
    end_ind = 0
    while start_ind < NUM_SAMPLES:
        num_remaining = NUM_SAMPLES - start_ind

        # sample a random file
        f_index = np.random.choice(len(im_filenames))
        im_data = np.load(os.path.join(DATASET_DIR, im_filenames[f_index]))['arr_0']
        pose_data = pose_preproc(np.load(os.path.join(DATASET_DIR, pose_filenames[f_index]))['arr_0'])
        
        # add datapoints
        num_datapoints_to_take = min(num_remaining, im_data.shape[0])
        end_ind = start_ind + num_datapoints_to_take
        im_tensor[start_ind:end_ind, ...] = im_data[:num_datapoints_to_take, ...]
        pose_tensor[start_ind:end_ind, ...] = pose_data[:num_datapoints_to_take]
        start_ind = end_ind
    
    # initialize GQCNN
    logging.info('Initializing GQCNN')
    gqcnn = get_gqcnn_model(BACKEND).load(MODEL_DIR)

    if BACKEND == 'tf':
        # open TF session
        gqcnn.open_session()
    
    # warmup GPU backend if necessary
    if GPU:
        logging.info('Warming up GPU')
        for i in range(GPU_WARM_ITERATIONS):
            start_time = time.time()
            pred = gqcnn.predict(im_tensor, pose_tensor)
            test_duration = time.time() - start_time
            logging.info('GPU warmup iteration {} took {} seconds'.format(i, test_duration))

    # run benchmark 
    logging.info('Beginning test')
    times = np.zeros((TEST_ITERATIONS,))
    for i in range(TEST_ITERATIONS):
        start_time = time.time()
        pred = gqcnn.predict(im_tensor, pose_tensor)
        test_duration = time.time() - start_time
        logging.info('Iteration {} took {} seconds'.format(i, test_duration))    
        times[i] = test_duration

    avg_duration = np.mean(times)
    logging.info('Average inference time was {} seconds over {} trials'.format(avg_duration, TEST_ITERATIONS))
    
    if BACKEND == 'tf':
        # close TF session
        gqcnn.close_session()

    if SAVE_PRED:
        logging.info('Saving predictions')
        np.savez_compressed(os.path.join(SAVE_DIR, SAVE_FNAME), pred)
    
    logging.info('Benchmark Finished')

