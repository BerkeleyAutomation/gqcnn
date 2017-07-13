"""
Script to generate T-SNE Visualization for a GQ-CNN model.
Author: Vishal Satish
"""

import logging
import os
import cPickle as pkl
import IPython
import numpy as np

from autolab_core import YamlConfig
from gqcnn import InputDataMode, ImageFileTemplates, Visualizer as vis2d

from bhtsne import tsne

def read_pose_data(pose_arr, input_data_mode):
    """ Read the pose data """
    if input_data_mode == InputDataMode.TF_IMAGE:
        return pose_arr[:,2:3]
    elif input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
        return np.c_[pose_arr[:,2:3], pose_arr[:,4:6]]
    elif input_data_mode == InputDataMode.RAW_IMAGE:
        return pose_arr[:,:4]
    elif input_data_mode == InputDataMode.RAW_IMAGE_PERSPECTIVE:
        return pose_arr[:,:6]
    elif input_data_mode == InputDataMode.REGRASPING:
        # depth, approach angle, and delta angle for reorientation
        return np.c_[pose_arr[:,2:3], pose_arr[:,4:5], pose_arr[:,6:7]]
    else:
        raise ValueError('Input data mode %s not supported' %(input_data_mode))

if __name__ == '__main__':
    # setup logger
    logging.getLogger().setLevel(logging.INFO)

    # get configuration parameters
    logging.info('Reading in T-SNE Generation Parameters.')
    config = YamlConfig('cfg/tools/generate_gqcnn_tsne.yaml')

    dataset_dir = config['dataset_dir']
    feature_dir = config['feature_dir']
    output_dir = config['output_dir']
    
    layers = config['layers']
    pre_computed_layers = config['pre_computed_layers']
    
    image_mode = config['image_mode']
    pose_data_format = config['pose_data_format']
    
    metric_thresh = config['metric_thresh']
    metric_name = config['metric_name']

    plotting_config = config['plotting']

    figsize = plotting_config['figsize']
    font_size = plotting_config['font_size']
    point_size = plotting_config['point_size']
    dpi = plotting_config['dpi']

    # load in feature file indices
    feature_generation_indices = pkl.load(open(os.path.join(feature_dir, 'feature_generation_indices.pkl'), 'rb'))
    feature_keys = feature_generation_indices.keys()    
    feature_keys.sort(key = lambda x: int(x[-9:-4]))
    
    # allocate label tensor
    num_datapoints = len(feature_generation_indices) * len(feature_generation_indices[feature_generation_indices.keys()[0]])
    labels = np.zeros(num_datapoints)
    poses = None

    # iterate through the file indices and pull out the necessary label and pose data
    logging.info('Loading label and pose data')
    start_ind = 0
    for filename in feature_keys:
        
        # extract file number
        file_number = filename[-9:-4]
        logging.info('Loading poses and labels for file {}'.format(file_number))

        # generate pose and metric filenames 
        pose_filename = (ImageFileTemplates.hand_poses_template + "_{}.npz").format(file_number)
        metric_filename = (metric_name + "_{}.npz").format(file_number)

        pose_tensor = read_pose_data(np.load(os.path.join(dataset_dir, pose_filename))['arr_0'][feature_generation_indices[filename]], pose_data_format)
        metric_tensor = np.load(os.path.join(dataset_dir, metric_filename))['arr_0'][feature_generation_indices[filename]]
        # print(metric_tensor)
        label_tensor = 1 * (metric_tensor > metric_thresh)
        # IPython.embed()

        if poses is None:
            poses = np.zeros((num_datapoints, pose_tensor.shape[1]))

        # populate pose and label tensors
        end_ind = start_ind + pose_tensor.shape[0]
        poses[start_ind:end_ind] = pose_tensor
        labels[start_ind:end_ind] = label_tensor

        start_ind = end_ind


    # generate T-SNE features
    logging.info('Generating T-SNE Features')
    for layer in layers:
        # check if layer has already been computed
        if not layer in pre_computed_layers:
            # load in the layer features
            logging.info('Loading features for layer {}'.format(layer))
            feature_tensor_name = 'features_{}.npz'.format(layer)
            feature_tensor = np.load(os.path.join(feature_dir, feature_tensor_name))['arr_0'][:len(poses)]

            # add in pose data if necessary
            if layer == 'fc4' or layer == 'fc5':
                full_feature_tensor = feature_tensor
            else:
                full_feature_tensor = np.c_[feature_tensor, poses]

            # perform t-sne
            logging.info('Performing T-SNE on layer {}'.format(layer))
            tsne_features = tsne(full_feature_tensor)

            # save features 
            logging.info('Saving Features')
            tsne_features_filename = os.path.join(output_dir, 'tsne_features_{}.npz'.format(layer))
            np.savez_compressed(tsne_features_filename, tsne_features)
        else:
            # load pre-computed tsne data
            logging.info('Found pre-computed t-sne data for layer {}'.format(layer))
            tsne_features = np.load(os.path.join(output_dir, 'tsne_features_{}.npz'.format(layer)))['arr_0']

        # generate plots
        logging.info('Generating T-SNE Plot for layer {}'.format(layer))
        vis2d.figure(size=(figsize,figsize))
        vis2d.scatter(tsne_features[labels==0, 0], tsne_features[labels==0, 1], s=point_size, c='r', label='Failures')
        vis2d.scatter(tsne_features[labels==1, 0], tsne_features[labels==1, 1], s=point_size, c='b', label='Successes')
        vis2d.xlabel('T-SNE Dim 0', fontsize=font_size)
        vis2d.ylabel('T-SNE Dim 1', fontsize=font_size)
        vis2d.title('T-SNE for {}'.format(layer), fontsize=font_size)
        # vis2d.legend(loc='best')
        vis2d.savefig(os.path.join(output_dir, 'tsne_{}.pdf'.format(layer)), dpi=dpi)