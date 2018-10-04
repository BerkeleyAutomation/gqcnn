"""
Compute neural network features and visualize with t-sne
Author: Jeff Mahler
"""
import argparse
import copy
import cPickle as pkl
import gc
import IPython
import logging
import numpy as np
import os
import sys
import time

import scipy.sparse as ssp
import sklearn.decomposition as skd
import sklearn.manifold as skm

import autolab_core.utils as utils
from autolab_core import TensorDataset, YamlConfig
from gqcnn.model import get_gqcnn_model
from gqcnn.utils import ImageMode, TrainingMode, GripperMode, GeneralConstants, read_pose_data

from perception import BinaryImage, ColorImage, DepthImage, GdImage, GrayscaleImage, RgbdImage, RenderMode
from visualization import Visualizer2D as vis2d
from visualization import Visualizer3D as vis3d

if __name__ == '__main__':
    # parse args
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, default='cfg/tools/featurize_grasp_dataset.yaml', help='Config file for tensor featurization')
    args = parser.parse_args()
    config_filename = args.config_filename

    # read config
    config = YamlConfig(config_filename)
    tensor_dataset_dir = config['tensor_dataset']
    split_name = config['split_name']
    
    # read featurization params
    feature_layers = config['feature_layers']
    num_pca_components = config['num_pca_components']
    use_bhtsne = config['use_bhtsne']
    perplexity = config['perplexity']
    max_num_samples = config['max_num_samples']

    # read gqcnn input params
    image_type = config['image_type']
    gripper_mode = config['gripper_mode']
    metric_name = config['metric_name']
    metric_thresh = config['metric_thresh']

    # read plotting params
    figsize = config['figsize']
    point_size = config['point_size']
    font_size = config['font_size']

    # open dataset
    dataset = TensorDataset.open(tensor_dataset_dir)
    num_tensors = dataset.num_tensors
    num_datapoints = dataset.num_datapoints
    datapoints_per_file = dataset.datapoints_per_file
    train_indices, val_indices, _ = dataset.split(split_name)
    num_val_datapoints = val_indices.shape[0]

    ind = np.arange(num_val_datapoints)
    if val_indices.shape[0] > max_num_sample:
        ind = np.random.choice(num_val_datapoints, size=max_num_sample, replace=False).astype(np.uint32)
        val_indices = val_indices[ind]
        num_val_datapoints = val_indices.shape[0]

    # open output directory
    output_dir = os.path.join(tensor_dataset_dir, 'features')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # load a gqcnn
    logging.info('Loading GQ-CNN')
    model_dir = config['gqcnn']
    gqcnn = get_gqcnn_model().load(model_dir)
    gqcnn.open_session()

    # init feature maps
    start_i = 0
    feature_maps = {}
    for feature_layer in feature_layers:
        feature_maps[feature_layer] = None
    
    # iterate through tensors, featurizing each one
    tensor_indices = np.arange(num_tensors)
    poses = None
    labels = np.zeros(num_val_datapoints)
    for i in tensor_indices:
        logging.info('Featurizing tensor %d' %(i))

        # load indices
        datapoint_indices = dataset.datapoint_indices_for_tensor(i)
        val_index_subset = np.array([j for j in datapoint_indices if j in val_indices])
        val_index_subset = val_index_subset - np.min(datapoint_indices)        
        if val_index_subset.shape[0] == 0:
            continue

        # load data
        image_tensor = dataset.tensor(image_type, i).data[val_index_subset,...]
        grasps_tensor = dataset.tensor('grasps', i).data[val_index_subset,...]
        metric_tensor = dataset.tensor(metric_name, i).data[val_index_subset,...]
        label_tensor = 1 * (metric_tensor > metric_thresh)

        pose_tensor = read_pose_data(grasps_tensor, gqcnn._gripper_mode)

        # update end index
        end_i = start_i + image_tensor.shape[0]

        # store labels
        labels[start_i:end_i] = label_tensor
        
        # allocate poses if necessary
        if poses is None:
            poses = np.zeros((num_val_datapoints, pose_tensor.shape[1]))

        # store poses
        poses[start_i:end_i] = pose_tensor
        
        # featurize data
        for feature_layer in feature_layers:
            logging.info('Computing features for %s' %(feature_layer))

            # compute features
            features = gqcnn.featurize(image_tensor,
                                       pose_tensor,
                                       feature_layer=feature_layer)

            if config['vis']['features']:
                num_filters = features.shape[3]
                d = utils.sqrt_ceil(num_filters)
                
                vis2d.figure()
                for j in range(num_filters):
                    vis2d.subplot(d,d,j+1)
                    vis2d.imshow(DepthImage(features[0,:,:,j]))
                vis2d.show()
                
            # flatten
            features = features.reshape(features.shape[0], -1)

            # allocate if necessary
            if feature_maps[feature_layer] is None:
                feature_maps[feature_layer] = np.zeros((num_val_datapoints, features.shape[1]), dtype=np.float32)

            # flatten and add to feature arrays
            feature_maps[feature_layer][start_i:end_i] = features

        # update indices
        start_i = end_i

        # cleanup
        del image_tensor
        del grasps_tensor
        del pose_tensor
        del metric_tensor
        del label_tensor
        gc.collect()

    # save the indices
    indices_filename = os.path.join(output_dir, 'feature_dataset_indices.npz')
    np.savez_compressed(indices_filename, val_indices)
        
    # close session
    gqcnn.close_session()

    # save features to file
    if config['save_raw_features']:
        for feature_layer, features in feature_maps.iteritems():
            logging.info('Saving features for %s' %(feature_layer))
        
            # save to file
            features_filename = os.path.join(output_dir, 'features_%s.npz' %(feature_layer))
            np.savez_compressed(features_filename, features)

    # dimensionality reduction
    for feature_layer, features in feature_maps.iteritems():
        logging.info('Performing PCA on %s' %(feature_layer))

        # concatenate with poses
        if feature_layer == 'fc4' or feature_layer == 'fc5':
            all_features = features
        else:
            all_features = np.c_[features, poses]
        
        # perform truncated SVD (for sparsity)
        pca = skd.TruncatedSVD(n_components=num_pca_components)
        tf_features = pca.fit_transform(all_features)

        # save pca to file
        pca_filename = os.path.join(output_dir, 'pca_%s.pkl' %(feature_layer))
        pkl.dump(pca, open(pca_filename, 'wb'))

        # save to file
        pca_features_filename = os.path.join(output_dir, 'tf_features_%s.npz' %(feature_layer))
        np.savez_compressed(pca_features_filename, tf_features)
        
        # run t-sne
        logging.info('Running T-SNE for %s' %(feature_layer))
        if use_bhtsne:
            # bhtsne impl
            from bhtsne import run_bh_tsne
            bhtsne_start = time.time()
            tsne_features = run_bh_tsne(tf_features, initial_dims=all_features.shape[1], verbose=True, perplexity=perplexity)
            logging.info('BH T-SNE took %.3f sec' %(time.time() - bhtsne_start))
        else:
            # sklearn impl
            tsne_start = time.time()
            tsne = skm.TSNE(perplexity=perplexity,
                            verbose=1)
            tsne_features = tsne.fit_transform(tf_features)
            logging.info('Sklearn T-SNE took %.3f sec' %(time.time() - tsne_start))

            # save t-sne to file
            tsne_filename = os.path.join(output_dir, 'tsne_%s.pkl' %(feature_layer))
            pkl.dump(tsne, open(tsne_filename, 'wb'))

        # save features to file
        tsne_features_filename = os.path.join(output_dir, 'tsne_features_%s.npz' %(feature_layer))
        np.savez_compressed(tsne_features_filename, tsne_features)

        # save figure to file
        try:
            vis2d.figure(size=(figsize,figsize))
            vis2d.scatter(tsne_features[labels==0,0], tsne_features[labels==0,1], s=point_size, c='r', label='Failures')
            vis2d.scatter(tsne_features[labels==1,0], tsne_features[labels==1,1], s=point_size, c='b', label='Successes')
            vis2d.xlabel('T-SNE Dim 0', fontsize=font_size)
            vis2d.ylabel('T-SNE Dim 1', fontsize=font_size)
            vis2d.title('T-SNE for %s' %(feature_layer), fontsize=font_size)
            vis2d.legend(loc='best')
            vis2d.savefig(os.path.join(output_dir, 'tsne_%s.pdf' %(feature_layer)), dpi=dpi)   
        except:
            pass
