"""
Script to convert TensorDataset split indices to legacy indices format

Author: Vishal Satish
"""
import argparse
import logging
import cPickle as pkl
import os

import numpy as np

if __name__ == '__main__':
    # setup logger
    logging.getLogger().setLevel(logging.INFO)
 
    # parse args
    parser = argparse.ArgumentParser(description='Convert TensorDataset split indices to legacy indices format')
    parser.add_argument('tensor_dataset_path', type=str, default=None, help='Path to the TensorDataset')
    parser.add_argument('--image_file_prefix', type=str, default='tf_depth_ims', help='Prefix of image filenames')
    parser.add_argument('--split_type', type=str, default='image_wise', help='Type of split')
    args = parser.parse_args()

    split_dir = os.path.join(args.tensor_dataset_path, 'splits', args.split_type)
    tensor_dir = os.path.join(args.tensor_dataset_path, 'tensors')

    # load TensorDataset split indices
    logging.info('Loading {} split indices for TensorDataset {}...'.format(args.split_type, args.tensor_dataset_path))
    train_indices_td = np.load(os.path.join(split_dir, 'train_indices.npz'))['arr_0']
    val_indices_td = np.load(os.path.join(split_dir, 'val_indices.npz'))['arr_0']

    # load all image filenames and sort
    logging.info('Loading image filenames...')
    im_filenames = [f for f in os.listdir(tensor_dir) if f.find(args.image_file_prefix) > -1]
    im_filenames.sort(key=lambda x: int(x[-9:-4]))

    num_datapoints_per_file = np.load(os.path.join(tensor_dir, im_filenames[0]))['arr_0'].shape[0]

    # generate legacy indices dict
    logging.info('Converting indices...')
    train_indices_leg = {}
    val_indices_leg = {}
    
    start_i = 0
    for f in im_filenames:
        end_i = start_i + num_datapoints_per_file
        train_indices_leg[f] = train_indices_td[np.where(np.logical_and(train_indices_td >= start_i, train_indices_td < end_i))[0]] % num_datapoints_per_file
        val_indices_leg[f] = val_indices_td[np.where(np.logical_and(val_indices_td >= start_i, val_indices_td < end_i))[0]] % num_datapoints_per_file
        start_i = end_i

    # save legacy indices in same dir as TensorDataset split indices
    logging.info('Saving legacy indices...')
    with open(os.path.join(split_dir, 'train_indices_{}.pkl'.format(args.split_type)), 'wb') as fhandle:
        pkl.dump(train_indices_leg, fhandle)
    with open(os.path.join(split_dir, 'val_indices_{}.pkl'.format(args.split_type)), 'wb') as fhandle:
        pkl.dump(val_indices_leg, fhandle)
    
