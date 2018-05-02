"""
Script for extracting full images from a tensor dataset and saving in format similar to legacy GQCNN dataset.
Author: Vishal Satish
"""
import logging
import numpy as np
import os
import time
import IPython

import matplotlib.pyplot as plt

from autolab_core import TensorDataset
from perception import DepthImage

IM_DATASET_PATH = '/nfs/diskstation/vsatish/dex-net/data/datasets/fizzytablets_dexnet_thresh_image_10_states_10_images/images/'
IM_PER_FILE = 100
FNAME_PLACE = 6
IM_FILE_TEMPLATE = 'depth_ims_tf_table'
OUTPUT_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/fizzytablets_dexnet_thresh_full_ims_04_30_18/'
VIS = 0
DEBUG = 0

if __name__ == '__main__':
    # setup logger
    logging.getLogger().setLevel(logging.INFO)    

    # get start time
    gen_start_time = time.time()

    # open datasets
    logging.info('Opening image dataset')
    im_dataset = TensorDataset.open(IM_DATASET_PATH)
    
    im_buffer = None

    # iterate through the image dataset
    buffer_ind = 0
    out_file_idx = 0
    for im_idx, datum in enumerate(im_dataset):
        im = DepthImage(datum['depth_ims'])
        logging.info('Image {} of {}'.format(im_idx + 1, im_dataset.num_datapoints))

        # allocate buffer if necessary
        if im_buffer is None:
            logging.info('Allocating buffers')
            im_buffer = np.zeros((IM_PER_FILE,) + im.raw_data.shape)

        # vis image
        if VIS:
            plt.figure()
            plt.imshow(im.raw_data[..., 0], cmap=plt.cm.gray)
            plt.show()
                    
        # add to buffer
        im_buffer[buffer_ind, ...] = im.raw_data
        buffer_ind += 1            

        # write out when buffer is full
        if buffer_ind >= IM_PER_FILE:
            # dump IM_PER_FILE datums
            logging.info('Saving {} datapoints'.format(IM_PER_FILE))
            im_fname = '{}_{}'.format(IM_FILE_TEMPLATE, str(out_file_idx).zfill(FNAME_PLACE))
            np.savez_compressed(os.path.join(OUTPUT_DIR, im_fname), im_buffer[:IM_PER_FILE])
            out_file_idx += 1
            buffer_ind = 0
    logging.info('Dataset generation took {} seconds'.format(time.time() - gen_start_time))
