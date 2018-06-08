import os
import logging
import argparse

import numpy as np

IMS_PER_FILE = 100
METRIC_FILE_TEMPLATE = 'robust_wrench_resistance'
METRIC_THRESH = 0.75

# setup logger level
logging.getLogger().setLevel(logging.INFO)

# parse args
arg_parser = argparse.ArgumentParser(description='Calculate metrics for the given legacy GQCNN dataset')
arg_parser.add_argument('dataset_path', type=str, default=None, help='Path to dataset to calculate metrics for')
args = arg_parser.parse_args()
dataset_dir = args.dataset_path

# read filenames
logging.info('Reading filenames...')
all_filenames = os.listdir(dataset_dir)
metric_filenames = [fname for fname in all_filenames if fname.find(METRIC_FILE_TEMPLATE) > -1]
metric_filenames.sort(key=lambda x: int(x[-9:-4]))

# calculate general dataset metrics useful in other calculations
logging.info('Calculating general dataset metrics...')
# num datapoints in dataset
num_ims_last_file = np.load(os.path.join(dataset_dir, metric_filenames[-1]))['arr_0'].shape[0]
num_datapoints = (len(metric_filenames) - 1) * IMS_PER_FILE + num_ims_last_file
logging.info('Number of datapoints in dataset: {}'.format(num_datapoints))

# calculate % positives
logging.info('Calculating % positives...')
num_pos = 0
for fname in metric_filenames:
    metric_data = np.load(os.path.join(dataset_dir, fname))['arr_0']
    num_pos += np.where(metric_data > METRIC_THRESH)[0].shape[0]
logging.info('Percent positives in dataset: {}'.format(float(num_pos) / num_datapoints * 100))
