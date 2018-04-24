import os
import logging

import numpy as np

DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/salt_cube_leg_no_rot_04_23_18/'
IMS_PER_FILE = 100
METRIC_FILE_TEMPLATE = 'robust_wrench_resistance'
METRIC_THRESH = 0.75

# setup logger level
logging.getLogger().setLevel(logging.INFO)

# read filenames
logging.info('Reading filenames...')
all_filenames = os.listdir(DATASET_DIR)
metric_filenames = [fname for fname in all_filenames if fname.find(METRIC_FILE_TEMPLATE) > -1]
metric_filenames.sort(key=lambda x: int(x[-10:-4]))

# calculate general dataset metrics useful in other calculation
logging.info('Calculating general dataset metrics...')
# num datapoints in dataset
num_ims_last_file = np.load(os.path.join(DATASET_DIR, metric_filenames[-1]))['arr_0'].shape[0]
num_datapoints = (len(metric_filenames) - 1) * IMS_PER_FILE + num_ims_last_file
logging.info('Number of datapoints in dataset: {}'.format(num_datapoints))

# calculate % positives
logging.info('Calculating % positives')
num_pos = 0
for fname in metric_filenames:
    metric_data = np.load(os.path.join(DATASET_DIR, fname))['arr_0']
    num_pos += np.where(metric_data > METRIC_THRESH)[0].shape[0]
logging.info('Percent positives in dataset: {}'.format(float(num_pos) / num_datapoints * 100))
