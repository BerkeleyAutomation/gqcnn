"""
Script to generate T-SNE Enhanced T-SNE Visualization that contains
thumbnails of positives and negatives for a GQ-CNN model.
Author: Vishal Satish
"""

import logging
import os
import cPickle as pkl
import IPython
import numpy as np
import matplotlib.pyplot as plt

from autolab_core import YamlConfig
from gqcnn import InputDataMode, ImageFileTemplates
from visualization import Visualizer2D as vis2d

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
    logging.info('Reading in Enhanced T-SNE Generation Parameters.')
    config = YamlConfig('cfg/tools/generate_gqcnn_tsne_enhanced.yaml')

    dataset_dir = config['dataset_dir']
    tsne_feature_dir = config['tsne_feature_dir']
    output_dir = config['output_dir']
    feature_indices_dir = config['feature_indices_dir']
    
    layers = config['layers']
     
    image_mode = config['image_mode']
    pose_data_format = config['pose_data_format']
    
    metric_thresh = config['metric_thresh']
    metric_name = config['metric_name']

    plotting_config = config['plotting']

    figsize = plotting_config['figsize']
    font_size = plotting_config['font_size']
    point_size = plotting_config['point_size']
    dpi = plotting_config['dpi']
    subsample_ratio = plotting_config['subsample_ratio']
    eps = plotting_config['eps']

    # load in feature file indices
    feature_generation_indices = pkl.load(open(os.path.join(feature_indices_dir, 'feature_generation_indices.pkl'), 'rb'))
    feature_keys = feature_generation_indices.keys()    

    # sort the keys so there will be a one-to-one mapping with the features
    feature_keys.sort(key = lambda x: int(x[-9:-4]))
    
    # allocate label tensor
    num_datapoints = len(feature_generation_indices) * len(feature_generation_indices[feature_generation_indices.keys()[0]])
    labels = np.zeros(num_datapoints)
    
    poses = None
    images = None

    # iterate through the file indices and pull out the necessary label, pose, and image data
    logging.info('Loading label, pose, and image data')
    start_ind = 0
    for filename in feature_keys:
        
        # extract file number
        file_number = filename[-9:-4]
        logging.info('Loading poses, labels, and images for file {}'.format(file_number))

        # generate pose, metric, and image filenames 
        pose_filename = (ImageFileTemplates.hand_poses_template + "_{}.npz").format(file_number)
        metric_filename = (metric_name + "_{}.npz").format(file_number)
        image_filename = filename

        pose_tensor = read_pose_data(np.load(os.path.join(dataset_dir, pose_filename))['arr_0'][feature_generation_indices[filename]], pose_data_format)
        metric_tensor = np.load(os.path.join(dataset_dir, metric_filename))['arr_0'][feature_generation_indices[filename]]
        image_tensor = np.load(os.path.join(dataset_dir, image_filename))['arr_0'][feature_generation_indices[filename]]
        
        label_tensor = 1 * (metric_tensor > metric_thresh)
        
        # allocate the image and pose tensors if they are None
        if poses is None:
            poses = np.zeros((num_datapoints, pose_tensor.shape[1]))
        if images is None:
            images = np.zeros((num_datapoints, image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]))

        # populate pose, label, and image tensors
        end_ind = start_ind + pose_tensor.shape[0]
        poses[start_ind:end_ind] = pose_tensor
        labels[start_ind:end_ind] = label_tensor
        images[start_ind:end_ind] = image_tensor
        start_ind = end_ind

    # plot
    logging.info('Beginning Plotting')
    for layer in layers:
        logging.info('Plotting layer {}'.format(layer))
        
        # load features
        logging.info('Loading T-SNE features for layer {}'.format(layer))
        tsne_features = np.load(os.path.join(tsne_feature_dir, 'tsne_features_{}.npz'.format(layer)))['arr_0']

        # subsample
        logging.info('Sub-sampling datapoints')
        num_sample = tsne_features.shape[0] / subsample_ratio
        logging.info('Sub-sample size is {}'.format(num_sample))
        subsampled_indices = np.random.choice(tsne_features.shape[0], size=num_sample, replace=False)
        tsne_features = tsne_features[subsampled_indices]
        sub_sampled_images = images[subsampled_indices]
        sub_sampled_labels = labels[subsampled_indices]
        pos_indices = np.where(sub_sampled_labels==1)[0]
        neg_indices = np.where(sub_sampled_labels==0)[0]
        
        # begin plotting
        vis2d.figure(size=(figsize,figsize))
        ax = plt.gca()
        h = 1.0
        w = 1.0
        hp = 1.2
        wp = 1.2

        # negative example background
        neg_im = np.zeros([sub_sampled_images.shape[1], sub_sampled_images.shape[2], 3])
        neg_im[:,:,0] = 192 * np.ones([sub_sampled_images.shape[1], sub_sampled_images.shape[2]])
        neg_im = neg_im.astype(np.uint8)

        # positive example background
        pos_im = np.zeros([sub_sampled_images.shape[1], sub_sampled_images.shape[2], 3])
        pos_im[:,:,1] = 224 * np.ones([sub_sampled_images.shape[1], sub_sampled_images.shape[2]])
        pos_im = pos_im.astype(np.uint8)

        # plot negatives
        for neg_ind in neg_indices:
            tsne_feature = tsne_features[neg_ind,:]
            image = sub_sampled_images[neg_ind,:,:,0]

            extent = np.array([tsne_feature[0] - wp, tsne_feature[0] + wp,
                               tsne_feature[1] - hp, tsne_feature[1] + hp])
            ax.imshow(neg_im, extent=extent)

            extent = np.array([tsne_feature[0] - w, tsne_feature[0] + w,
                               tsne_feature[1] - h, tsne_feature[1] + h])
            ax.imshow(image, extent=extent, cmap=plt.cm.gray_r)

        # plot positives
        for pos_ind in pos_indices:
            tsne_feature = tsne_features[pos_ind,:]
            image = sub_sampled_images[pos_ind,:,:,0]

            extent = np.array([tsne_feature[0] - wp, tsne_feature[0] + wp,
                               tsne_feature[1] - hp, tsne_feature[1] + hp])
            ax.imshow(pos_im, extent=extent)

            extent = np.array([tsne_feature[0] - w, tsne_feature[0] + w,
                               tsne_feature[1] - h, tsne_feature[1] + h])
            ax.imshow(image, extent=extent, cmap=plt.cm.gray_r)

        min_tsne_dim_0 = np.min(tsne_features[:,0])
        max_tsne_dim_0 = np.max(tsne_features[:,0])
        min_tsne_dim_1 = np.min(tsne_features[:,1])
        max_tsne_dim_1 = np.max(tsne_features[:,1])
        vis2d.xlabel('T-SNE Dim 0', fontsize=font_size)
        vis2d.ylabel('T-SNE Dim 1', fontsize=font_size)
        vis2d.title('T-SNE for %s' %(layer), fontsize=font_size)
        ax.set_xlim(min_tsne_dim_0-eps, max_tsne_dim_0+eps)
        ax.set_ylim(min_tsne_dim_1-eps, max_tsne_dim_1+eps)  
        vis2d.savefig(os.path.join(output_dir, 'tsne_enhanced_%s.pdf' %(layer)), dpi=dpi)