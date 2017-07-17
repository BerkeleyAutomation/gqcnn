"""
Script to generate feature tensors for the validation data from a given dataset
using a given model.
Author: Vishal Satish
"""

from gqcnn import GQCNN, ImageMode, ImageFileTemplates, InputDataMode, Visualizer as vis2d
from autolab_core import YamlConfig
import autolab_core.utils as utils
from perception import DepthImage

import os
import gc 
import logging
import IPython
import cPickle as pickle
import numpy as np

import sklearn.decomposition as skd

def read_pose_data(pose_arr, input_data_mode):
    """ Read the pose data and slice it according to the specified input_data_mode

    Parameters
    ----------
    pose_arr: :obj:`ndArray`
        full pose data array read in from file
    input_data_mode: :obj:`InputDataMode`
        enum for input data mode, see optimizer_constants.py for all
        possible input data modes 

    Returns
    -------
    :obj:`ndArray`
        sliced pose_data corresponding to input data mode
    """
    if input_data_mode == InputDataMode.TF_IMAGE:
        return pose_arr[:,2:3]
    elif input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
        return np.c_[pose_arr[:,2:3], pose_arr[:,4:6]]
    elif input_data_mode == InputDataMode.RAW_IMAGE:
        return pose_arr[:,:4]
    elif input_data_mode == InputDataMode.RAW_IMAGE_PERSPECTIVE:
        return pose_arr[:,:6]
    else:
        raise ValueError('Input data mode {} not supported'.format(input_data_mode))

if __name__ == "__main__":
    # setup logger
    logging.getLogger().setLevel(logging.INFO)

    # read config parameters
    logging.info('Reading in featurization configurations.')
    cfg = YamlConfig('cfg/tools/generate_gqcnn_features.yaml')
    dataset_dir = cfg['dataset_dir']
    model_dir = cfg['model_dir']
    output_dir = cfg['output_dir']
    layers = cfg['layers']
    image_mode = cfg['image_mode']
    data_split = cfg['data_split']
    pose_data_format = cfg['pose_data_format']
    max_features = cfg['max_features']
    save_raw_features = cfg['save_raw_features']
    num_pca_components = cfg['num_pca_components']
    debug = cfg['debug']
    debug_num_files = cfg['debug_num_files']

    # read image filenames
    logging.info('Reading filenames')
    all_filenames = os.listdir(dataset_dir)
    if image_mode== ImageMode.BINARY:
        im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.binary_im_tensor_template) > -1]
    elif image_mode== ImageMode.DEPTH:
        im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.depth_im_tensor_template) > -1]
    elif image_mode== ImageMode.BINARY_TF:
        im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.binary_im_tf_tensor_template) > -1]
    elif image_mode== ImageMode.COLOR_TF:
        im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.color_im_tf_tensor_template) > -1]
    elif image_mode== ImageMode.GRAY_TF:
        im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.gray_im_tf_tensor_template) > -1]
    elif image_mode== ImageMode.DEPTH_TF:
        im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.depth_im_tf_tensor_template) > -1]
    elif image_mode== ImageMode.DEPTH_TF_TABLE:
        im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.depth_im_tf_table_tensor_template) > -1]
    else:
        raise ValueError('Image mode {} not supported.'.format(image_mode))
    
    # read pose filenames
    pose_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.hand_poses_template) > -1]

    # sort the image and pose filenames so they match sequentially  
    im_filenames.sort(key = lambda x: int(x[-9:-4]))
    pose_filenames.sort(key = lambda x: int(x[-9:-4]))

    # if debugging only sample a certain number of files
    if debug:
        im_filenames = im_filenames[:debug_num_files]
        pose_filenames = pose_filenames[:debug_num_files]

    # get the validation indices used for training the model
    if data_split == 'image_wise':
        validation_file_indices = pickle.load(open(os.path.join(model_dir, 'val_indices_image_wise.pkl'), 'rb'))
    elif data_split == 'stable_pose_wise':
        validation_file_indices = pickle.load(open(os.path.join(model_dir, 'val_indices_stable_pose_wise.pkl'), 'rb'))
    elif data_split == 'object_wise':
        validation_file_indices = pickle.load(open(os.path.join(model_dir, 'val_indices_object_wise.pkl'), 'rb'))
    else:
        raise ValueError('Data split mode {} not supported.'.format(data_split))

    # remove files than have no validation indices from the dictionary
    validation_file_indices = {key: value for key, value in validation_file_indices.iteritems() if len(value) > 0}
    
    # sub-sample to max_features
    num_samples_per_file = np.floor(max_features / len(validation_file_indices))
    logging.info('Sampling {} datapoints per file.'.format(int(num_samples_per_file)))
 
    for key in validation_file_indices:
        validation_file_indices[key] = np.random.choice(validation_file_indices[key], int(np.min((len(validation_file_indices[key]), num_samples_per_file))), replace=False)
   
    # save the file indices used for feature generation
    with open(os.path.join(output_dir, 'feature_generation_indices.pkl'), 'wb') as output_file:
        pickle.dump(validation_file_indices, output_file)

    # load the model into a GQ-CNN
    gqcnn = GQCNN.load(model_dir)
    gqcnn.open_session()
    logging.info('Created GQ-CNN Model')

    # iterate through each layer and generate the feature maps for it
    for layer in layers:
        logging.info('Generating feature maps for layer {}'.format(layer))

        # create tensor to hold features
        feature_arr = None
        
        start_i = 0
        # load each image file and its corresponding pose file if there are validation indices in it
        for x in range(len(im_filenames)):
            filename = im_filenames[x]
            if filename in validation_file_indices.keys():
                image_arr = np.load(os.path.join(dataset_dir, filename))['arr_0'][validation_file_indices[filename]]
                pose_arr = read_pose_data(np.load(os.path.join(dataset_dir, pose_filenames[x]))['arr_0'][validation_file_indices[filename]], pose_data_format)
                
                # pull features from the GQ-CNN
                logging.info('Computing feature maps for layer {} with image file {}'.format(layer, filename))
                feature_maps = gqcnn.featurize(image_arr, pose_arr, layer)[:len(image_arr)]
                
                # visualize feature maps if specified
                if cfg['vis']['features']:
                    num_filters = feature_maps.shape[3]
                    d = utils.sqrt_ceil(num_filters)
                
                    vis2d.figure()
                    for j in range(num_filters):
                        vis2d.subplot(d,d,j+1)
                        vis2d.imshow(DepthImage(feature_maps[0,:,:,j]))
                    vis2d.show()
                
                # flatten two-dimensional feature maps into one-dimensional array
                flattened_maps = feature_maps.reshape((feature_maps.shape[0], -1))

                # update the end index
                end_i = start_i + len(flattened_maps)

                # allocate a new tensor for this layer in the dict if necessary
                if feature_arr is None:
                    feature_arr = np.zeros((max_features, flattened_maps.shape[1])) 
                
                # add to feature map arr
                feature_arr[start_i:end_i] = flattened_maps
                
                # update the start index
                start_i = end_i

        # we allocated feature_arr to have max_features size but it might not be full
        feature_arr = feature_arr[:end_i]

        # save the raw features before PCA if specified in config
        if save_raw_features:
            logging.info('Saving features for layer {}'.format(layer))
            output_filename = os.path.join(output_dir, 'features_{}.npz'.format(layer))
            np.savez_compressed(output_filename, feature_arr)

        # perform PCA transform
        logging.info('Performing PCA on layer {}'.format(layer))

        # perform truncated SVD (for sparsity)
        pca = skd.TruncatedSVD(n_components=num_pca_components)
        transformed_features = pca.fit_transform(feature_arr)

        # save pca to file
        pca_filename = os.path.join(output_dir, 'pca_{}.pkl'.format(layer))
        pickle.dump(pca, open(pca_filename, 'w'))

        # save to file
        pca_features_filename = os.path.join(output_dir, 'tf_features_{}.npz'.format(layer))
        np.savez_compressed(pca_features_filename, transformed_features)

        # clean-up to clear-up memory 
        del feature_arr
        del image_arr
        del pose_arr
        del flattened_maps
        gc.collect()

    # close the gqcnn session
    gqcnn.close_session()
    

