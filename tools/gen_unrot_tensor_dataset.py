"""
Script for generating un-rotated training images for an existing TensorDataset.
Author: Vishal Satish
"""
import logging
import os
import time
import shutil
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from autolab_core import TensorDataset
from perception import DepthImage
from visualization import Visualizer2D as vis2d

IM_FILE_TEMPLATE = 'tf_depth_ims_no_rot'
ORIG_IM_FILE_TEMPLATE = 'tf_depth_ims'
VIS = 0

if __name__ == '__main__':
    # setup logger
    logging.getLogger().setLevel(logging.INFO)    

    # parse args
    parser = argparse.ArgumentParser(description='Generate un-rotated training images for an existing TensorDataset')
    parser.add_argument('tensor_dataset_path', type=str, default=None, help='Path to the TensorDataset containing image and grasp datasets')
    args = parser.parse_args()
    im_dataset_path = os.path.join(args.tensor_dataset_path, 'images')
    grasp_dataset_path = os.path.join(args.tensor_dataset_path, 'grasps')
    output_dir = os.path.join(grasp_dataset_path, 'tensors')

    # get start time
    gen_start_time = time.time()

    # open datasets
    logging.info('Opening datasets...')
    im_dataset = TensorDataset.open(im_dataset_path)
    grasp_dataset = TensorDataset.open(grasp_dataset_path)

    # read metadata
    ims_per_file = grasp_dataset.datapoints_per_file
    im_config = grasp_dataset.config['fields'][ORIG_IM_FILE_TEMPLATE]    
    train_im_h = im_config['height'] 
    train_im_w = im_config['width']
    train_im_num_channels = im_config['channels']
    fname_place = grasp_dataset.filename_numeric_label_place

    # generate buffers
    logging.info('Allocating buffers...')
    im_buffer = np.zeros((ims_per_file, train_im_h, train_im_w, train_im_num_channels))

    # iterate through the image dataset
    buffer_ind = 0
    out_file_idx = 0
    for im_idx, datum in enumerate(im_dataset):
        im = DepthImage(datum['depth_ims'])
        grasp_start_ind = datum['grasp_start_ind']
        grasp_end_ind = datum['grasp_end_ind']

        # iterate through the corresponding grasps
        for i in range(grasp_start_ind, grasp_end_ind):
            grasp = grasp_dataset[i]
            pose = grasp['grasps']            
            
            # align grasp
            logging.info('Aligning grasp {} of {} for image {} of {}'.format(i + 1 - grasp_start_ind, grasp_end_ind - grasp_start_ind, im_idx + 1, im_dataset.num_datapoints))
            
            # center & crop
            tf_im = im.align(1.0, np.asarray([pose[1], pose[0]]), 0.0, train_im_h, train_im_w)

            # vis original bin image and crop
            if VIS:
                logging.info('Crop X: {}, Crop Y: {}'.format(pose[1], pose[0]))
                plt.figure()
                plt.subplot(121)
                plt.imshow(im.raw_data[..., 0], cmap=plt.cm.gray)
                plt.subplot(122)
                plt.imshow(tf_im.raw_data[..., 0], cmap=plt.cm.gray)
                plt.show()
            
            # add to buffers
            im_buffer[buffer_ind, ...] = tf_im.raw_data
            buffer_ind += 1            

            # write out when buffers are full
            if buffer_ind >= ims_per_file:
                # dump IM_PER_FILE datums
                logging.info('Saving {} datapoints'.format(ims_per_file))
                im_fname = '{}_{}'.format(IM_FILE_TEMPLATE, str(out_file_idx).zfill(fname_place))
                np.savez_compressed(os.path.join(output_dir, im_fname), im_buffer[:ims_per_file])
                out_file_idx += 1
                im_buffer[:buffer_ind % ims_per_file] = im_buffer[ims_per_file:buffer_ind]
                buffer_ind = buffer_ind % ims_per_file

    # write out once at the end
    logging.info('Saving {} datapoints'.format(buffer_ind))
    im_fname = '{}_{}'.format(IM_FILE_TEMPLATE, str(out_file_idx).zfill(fname_place))
    np.savez_compressed(os.path.join(output_dir, im_fname), im_buffer[:buffer_ind])
 
    # update dataset metadata
    logging.info('Updating dataset metadata...')
    metadata_file = os.path.join(grasp_dataset_path, 'config.json')
    with open(metadata_file, 'rb') as fhandle:
        metadata = json.load(fhandle)
    metadata['fields'][IM_FILE_TEMPLATE] = metadata['fields'][ORIG_IM_FILE_TEMPLATE]
    with open(metadata_file, 'wb') as fhandle:
        json.dump(metadata, fhandle)

    logging.info('Dataset generation took {} seconds'.format(time.time() - gen_start_time))
