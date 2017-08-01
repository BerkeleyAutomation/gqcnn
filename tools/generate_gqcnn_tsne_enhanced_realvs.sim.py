"""
Script to generate feature tensors for the validation data from a given dataset
and data from a dataset from the real robot and then compute an enhanced thumbnail T-SNE plot
for the combined data. 
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
from bhtsne import tsne
import matplotlib.pyplot as plt

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
    cfg = YamlConfig('cfg/tools/generate_gqcnn_tsne_enhanced_realvs.sim.yaml')
    use_pre_computed_features = cfg['use_pre_computed_features']
    featurization_cfg = YamlConfig('cfg/tools/generate_gqcnn_tsne_enhanced_realvs.sim.yaml')['featurization_config']
    tsne_cfg = YamlConfig('cfg/tools/generate_gqcnn_tsne_enhanced_realvs.sim.yaml')['tsne_config']
    real_dataset_dir = featurization_cfg['real_dataset_dir']
    sim_dataset_dir = featurization_cfg['sim_dataset_dir']
    model_dir = featurization_cfg['model_dir']
    output_dir = featurization_cfg['output_dir']
    layers = featurization_cfg['layers']
    image_mode = featurization_cfg['image_mode']
    data_split = featurization_cfg['data_split']
    pose_data_format = featurization_cfg['pose_data_format']
    save_raw_features = featurization_cfg['save_raw_features']
    num_pca_components = featurization_cfg['num_pca_components']
    debug = featurization_cfg['debug']
    debug_num_files = featurization_cfg['debug_num_files']

    metric_thresh = tsne_cfg['metric_thresh']
    metric_name = tsne_cfg['metric_name']

    plotting_config = tsne_cfg['plotting']

    figsize = plotting_config['figsize']
    font_size = plotting_config['font_size']
    point_size = plotting_config['point_size']
    dpi = plotting_config['dpi']
    subsample_ratio = plotting_config['subsample_ratio']
    eps = plotting_config['eps']

    ### PART 1: FEATURIZATION OF REAL AND SIMULATED DATSETS ###
    logging.info('FEATURIZING')
    if not use_pre_computed_features:
        # read image filenames for real dataset
        logging.info('Reading filenames for simulated dataset')
        all_simulated_filenames = os.listdir(sim_dataset_dir)
        if image_mode == ImageMode.BINARY:
            sim_im_filenames = [f for f in all_simulated_filenames if f.find(ImageFileTemplates.binary_im_tensor_template) > -1]
        elif image_mode == ImageMode.DEPTH:
            sim_im_filenames = [f for f in all_simulated_filenames if f.find(ImageFileTemplates.depth_im_tensor_template) > -1]
        elif image_mode == ImageMode.BINARY_TF:
            sim_im_filenames = [f for f in all_simulated_filenames if f.find(ImageFileTemplates.binary_im_tf_tensor_template) > -1]
        elif image_mode == ImageMode.COLOR_TF:
            sim_im_filenames = [f for f in all_simulated_filenames if f.find(ImageFileTemplates.color_im_tf_tensor_template) > -1]
        elif image_mode == ImageMode.GRAY_TF:
            sim_im_filenames = [f for f in all_simulated_filenames if f.find(ImageFileTemplates.gray_im_tf_tensor_template) > -1]
        elif image_mode == ImageMode.DEPTH_TF:
            sim_im_filenames = [f for f in all_simulated_filenames if f.find(ImageFileTemplates.depth_im_tf_tensor_template) > -1]
        elif image_mode == ImageMode.DEPTH_TF_TABLE:
            sim_im_filenames = [f for f in all_simulated_filenames if f.find(ImageFileTemplates.depth_im_tf_table_tensor_template) > -1]
        else:
            raise ValueError('Image mode {} not supported.'.format(image_mode))
        
        # read pose filenames
        sim_pose_filenames = [f for f in all_simulated_filenames if f.find(ImageFileTemplates.hand_poses_template) > -1]

        # sort the image and pose filenames so they match sequentially  
        sim_im_filenames.sort(key = lambda x: int(x[-9:-4]))
        sim_pose_filenames.sort(key = lambda x: int(x[-9:-4]))

        # if debugging only sample a certain number of files
        if debug:
            sim_im_filenames = sim_im_filenames[:debug_num_files]
            sim_pose_filenames = sim_pose_filenames[:debug_num_files]

        logging.info('Loading validation indices')
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
        
        # read image filenames for real dataset
        logging.info('Reading filenames for real dataset')
        all_real_filenames = os.listdir(real_dataset_dir)
        if image_mode == ImageMode.BINARY:
            real_im_filenames = [f for f in all_real_filenames if f.find(ImageFileTemplates.binary_im_tensor_template) > -1]
        elif image_mode == ImageMode.DEPTH:
            real_im_filenames = [f for f in all_real_filenames if f.find(ImageFileTemplates.depth_im_tensor_template) > -1]
        elif image_mode == ImageMode.BINARY_TF:
            real_im_filenames = [f for f in all_real_filenames if f.find(ImageFileTemplates.binary_im_tf_tensor_template) > -1]
        elif image_mode == ImageMode.COLOR_TF:
            real_im_filenames = [f for f in all_real_filenames if f.find(ImageFileTemplates.color_im_tf_tensor_template) > -1]
        elif image_mode == ImageMode.GRAY_TF:
            real_im_filenames = [f for f in all_real_filenames if f.find(ImageFileTemplates.gray_im_tf_tensor_template) > -1]
        elif image_mode == ImageMode.DEPTH_TF:
            real_im_filenames = [f for f in all_real_filenames if f.find(ImageFileTemplates.depth_im_tf_tensor_template) > -1]
        elif image_mode == ImageMode.DEPTH_TF_TABLE:
            real_im_filenames = [f for f in all_real_filenames if f.find(ImageFileTemplates.depth_im_tf_table_tensor_template) > -1]
        else:
            raise ValueError('Image mode {} not supported.'.format(image_mode))
        
        # read pose filenames
        real_pose_filenames = [f for f in all_real_filenames if f.find(ImageFileTemplates.hand_poses_template) > -1]

        # sort the image and pose filenames so they match sequentially  
        real_im_filenames.sort(key = lambda x: int(x[-9:-4]))
        real_pose_filenames.sort(key = lambda x: int(x[-9:-4]))

        # if debugging only sample a certain number of files
        if debug:
            real_im_filenames = real_im_filenames[:debug_num_files]
            real_pose_filenames = real_pose_filenames[:debug_num_files]
        
        # generate file indices for real dataset
        real_file_indices = {}
        for im_file in real_im_filenames:
            image_tensor = np.load(os.path.join(real_dataset_dir, im_file))['arr_0']
            real_file_indices[im_file] = np.arange(len(image_tensor))

        # now subsample a sample of size s from the simulated data where s=size of real data
        logging.info('Sub-sampling simulated data to match real data size')
        real_dataset_size = 0
        for im_file in real_im_filenames:
            image_tensor = np.load(os.path.join(real_dataset_dir, im_file))['arr_0']
            real_dataset_size += len(image_tensor)

        percentage_files_to_use = 1.0
        num_samples_per_file = real_dataset_size / float(len(validation_file_indices))
        if num_samples_per_file < 1:
            percentage_files_to_use = num_samples_per_file
            num_samples_per_file = 1

        subsampled_files = np.random.choice(validation_file_indices.keys(), int(percentage_files_to_use * len(validation_file_indices.keys())), replace=False)

        sim_file_indices = {}
        for file in subsampled_files:
            sim_file_indices[file] = np.random.choice(validation_file_indices[file], int(np.min((len(validation_file_indices[file]), num_samples_per_file))), replace=False)

        # save the file indices for the simulated and real data
        with open(os.path.join(output_dir, 'simulated_feature_generation_indices.pkl'), 'wb') as output_file:
            pickle.dump(sim_file_indices, output_file)
        with open(os.path.join(output_dir, 'real_feature_generation_indices.pkl'), 'wb') as output_file:
            pickle.dump(real_file_indices, output_file)

        # load the model into a GQ-CNN
        gqcnn = GQCNN.load(model_dir)
        gqcnn.open_session()
        logging.info('Created GQ-CNN Model')

        # sort the keys of the simulated feature generation indices and real feature generation indices
        sim_feature_generation_keys = sim_file_indices.keys()
        sim_feature_generation_keys.sort(key = lambda x: int(x[-9:-4]))
        real_feature_generation_keys = real_file_indices.keys()
        real_feature_generation_keys.sort(key = lambda x: int(x[-9:-4]))

        # create map to know which datapoints are real vs. simulated, 1 is real 0 is sim
        real_sim_map = np.zeros(2 * real_dataset_size)

        # iterate through each layer and generate the feature maps for it using the simulated and real data
        for layer in layers:
            logging.info('Generating feature maps for layer {}'.format(layer))

            # create tensor to hold features
            feature_arr = None
            
            start_i = 0

            # first add all the simulated files
            for filename in sim_feature_generation_keys:
                image_arr = np.load(os.path.join(sim_dataset_dir, filename))['arr_0'][sim_file_indices[filename]]
                pose_arr = read_pose_data(np.load(os.path.join(sim_dataset_dir, 'hand_poses_{}.npz'.format(filename[-9:-4])))['arr_0'][sim_file_indices[filename]], pose_data_format)
                
                # pull features from the GQ-CNN
                logging.info('Computing feature maps for layer {} with simulated image file {}'.format(layer, filename))
                feature_maps = gqcnn.featurize(image_arr, pose_arr, layer)[:len(image_arr)]
                
                # visualize feature maps if specified
                if featurization_cfg['vis']['features']:
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
                    feature_arr = np.zeros((2 * real_dataset_size, flattened_maps.shape[1])) 
                
                # add to feature map arr
                feature_arr[start_i:end_i] = flattened_maps

                # update real vs. sim map
                real_sim_map[start_i:end_i] = 0
                
                # update the start index
                start_i = end_i

            # next add all the real files
            for filename in real_feature_generation_keys:
                image_arr = np.load(os.path.join(real_dataset_dir, filename))['arr_0'][real_file_indices[filename]]
                pose_arr = read_pose_data(np.load(os.path.join(real_dataset_dir, 'hand_poses_{}.npz'.format(filename[-9:-4])))['arr_0'][real_file_indices[filename]], pose_data_format)
                
                # pull features from the GQ-CNN
                logging.info('Computing feature maps for layer {} with real image file {}'.format(layer, filename))
                feature_maps = gqcnn.featurize(image_arr, pose_arr, layer)[:len(image_arr)]
                
                # visualize feature maps if specified
                if featurization_cfg['vis']['features']:
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
                    feature_arr = np.zeros((2 * real_dataset_size, flattened_maps.shape[1])) 
                
                # add to feature map arr
                feature_arr[start_i:end_i] = flattened_maps
                
                # update real vs. sim map
                real_sim_map[start_i:end_i] = 1
                
                # update the start index
                start_i = end_i

            np.savez_compressed(os.path.join(output_dir, 'real_sim_map.npz'), real_sim_map)
            
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

    ### PART 2: ENHANCED T-SNE PLOT GENERATION ###
    logging.info('GENERATING T-SNE DATA')
    # load simulated indices, real indices, and real vs. sim map
    real_file_indices = pickle.load(open(os.path.join(output_dir, 'real_feature_generation_indices.pkl'), 'rb'))
    sim_file_indices = pickle.load(open(os.path.join(output_dir, 'simulated_feature_generation_indices.pkl'), 'rb'))
    real_sim_map = np.load(os.path.join(output_dir, 'real_sim_map.npz'))['arr_0']

    # sort the keys of the simulated feature generation indices and real feature generation indices
    sim_feature_generation_keys = sim_file_indices.keys()
    real_feature_generation_keys = real_file_indices.keys()

    # allocate label tensor
    num_datapoints = len(real_sim_map)
    labels = np.zeros(num_datapoints)
    poses = None
    images = None

    # load label and pose data for simulated data
    start_ind = 0
    logging.info('Loading label and pose data for simulated data')
    for filename in sim_feature_generation_keys:
        # extract file number
        file_number = filename[-9:-4]
        logging.info('Loading poses, labels, and images for simulated file {}'.format(file_number))

        # generate pose and metric filenames 
        pose_filename = (ImageFileTemplates.hand_poses_template + "_{}.npz").format(file_number)
        metric_filename = (metric_name + "_{}.npz").format(file_number)

        image_tensor = np.load(os.path.join(sim_dataset_dir, filename))['arr_0'][sim_file_indices[filename]]
        pose_tensor = read_pose_data(np.load(os.path.join(sim_dataset_dir, pose_filename))['arr_0'][sim_file_indices[filename]], pose_data_format)
        metric_tensor = np.load(os.path.join(sim_dataset_dir, metric_filename))['arr_0'][sim_file_indices[filename]]

        label_tensor = 1 * (metric_tensor > metric_thresh)

        if poses is None:
            poses = np.zeros((num_datapoints, pose_tensor.shape[1]))
        if images is None:
            images = np.zeros((num_datapoints, image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]))

        # populate pose and label tensors
        end_ind = start_ind + pose_tensor.shape[0]
        images[start_ind:end_ind] = image_tensor
        poses[start_ind:end_ind] = pose_tensor
        labels[start_ind:end_ind] = label_tensor

        start_ind = end_ind

    # load label and pose data for real data
    logging.info('Loading label and pose data for real data')
    for filename in real_feature_generation_keys:
        # extract file number
        file_number = filename[-9:-4]
        logging.info('Loading poses, labels, and images for real file {}'.format(file_number))

        # generate pose and metric filenames 
        pose_filename = (ImageFileTemplates.hand_poses_template + "_{}.npz").format(file_number)
        metric_filename = ("human_label" + "_{}.npz").format(file_number)

        image_tensor = np.load(os.path.join(real_dataset_dir, filename))['arr_0'][real_file_indices[filename]]
        pose_tensor = read_pose_data(np.load(os.path.join(real_dataset_dir, pose_filename))['arr_0'][real_file_indices[filename]], pose_data_format)
        metric_tensor = np.load(os.path.join(real_dataset_dir, metric_filename))['arr_0'][real_file_indices[filename]]

        if poses is None:
            poses = np.zeros((num_datapoints, pose_tensor.shape[1]))
        if images is None:
            images = np.zeros((num_datapoints, image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]))

        # populate pose and label tensors
        end_ind = start_ind + pose_tensor.shape[0]
        images[start_ind:end_ind] = image_tensor
        poses[start_ind:end_ind] = pose_tensor
        labels[start_ind:end_ind] = label_tensor

        start_ind = end_ind

    for layer in layers:
        # load in the layer features
        logging.info('Loading features for layer {}'.format(layer))
        feature_tensor_name = 'tf_features_{}.npz'.format(layer)
        feature_tensor = np.load(os.path.join(output_dir, feature_tensor_name))['arr_0'][:len(poses)]

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

    logging.info('Plotting T-SNE')
    for layer in layers:
        logging.info('Plotting layer {}'.format(layer))
        
        # load features
        logging.info('Loading T-SNE features for layer {}'.format(layer))
        tsne_features = np.load(os.path.join(output_dir, 'tsne_features_{}.npz'.format(layer)))['arr_0']

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
        real_indices = np.where(real_sim_map==1)[0]
        sim_indices = np.where(real_sim_map==0)[0]
        pos_real_indices = np.intersect1d(pos_indices, real_indices)
        neg_real_indices = np.intersect1d(neg_indices, real_indices)
        pos_sim_indices = np.intersect1d(pos_indices, sim_indices)
        neg_sim_indices = np.intersect1d(neg_indices, sim_indices)
        
        # begin plotting
        vis2d.figure(size=(figsize,figsize))
        ax = plt.gca()
        h = 1.0
        w = 1.0
        hp = 1.2
        wp = 1.2
        h_real_sim = 1.4
        w_real_sim = 1.4

        # negative example background, red
        neg_im = np.zeros([sub_sampled_images.shape[1], sub_sampled_images.shape[2], 3])
        neg_im[:,:,0] = 192 * np.ones([sub_sampled_images.shape[1], sub_sampled_images.shape[2]])
        neg_im = neg_im.astype(np.uint8)

        # positive example background, green
        pos_im = np.zeros([sub_sampled_images.shape[1], sub_sampled_images.shape[2], 3])
        pos_im[:,:,1] = 224 * np.ones([sub_sampled_images.shape[1], sub_sampled_images.shape[2]])
        pos_im = pos_im.astype(np.uint8)

        # real example background
        real_im = np.zeros([sub_sampled_images.shape[1], sub_sampled_images.shape[2], 3])
        real_im[:,:,0] = 600 * np.ones([sub_sampled_images.shape[1], sub_sampled_images.shape[2]])
        real_im = real_im.astype(np.uint8)

        # sim example background
        sim_im = np.zeros([sub_sampled_images.shape[1], sub_sampled_images.shape[2], 3])
        sim_im[:,:,0] = 7 * np.ones([sub_sampled_images.shape[1], sub_sampled_images.shape[2]])
        sim_im = sim_im.astype(np.uint8)

        # plot negative real images
        for neg_real_ind in neg_real_indices:
            tsne_feature = tsne_features[neg_real_ind,:]
            image = sub_sampled_images[neg_real_ind,:,:,0]

            extent = np.array([tsne_feature[0] - w_real_sim, tsne_feature[0] + w_real_sim,
                               tsne_feature[1] - h_real_sim, tsne_feature[1] + h_real_sim])
            ax.imshow(real_im, extent=extent)

            extent = np.array([tsne_feature[0] - wp, tsne_feature[0] + wp,
                               tsne_feature[1] - hp, tsne_feature[1] + hp])
            ax.imshow(neg_im, extent=extent)

            extent = np.array([tsne_feature[0] - w, tsne_feature[0] + w,
                               tsne_feature[1] - h, tsne_feature[1] + h])
            ax.imshow(image, extent=extent, cmap=plt.cm.gray_r)

        # plot positive real images
        for pos_real_ind in pos_real_indices:
            tsne_feature = tsne_features[pos_real_ind,:]
            image = sub_sampled_images[pos_real_ind,:,:,0]

            extent = np.array([tsne_feature[0] - w_real_sim, tsne_feature[0] + w_real_sim,
                               tsne_feature[1] - h_real_sim, tsne_feature[1] + h_real_sim])
            ax.imshow(real_im, extent=extent)

            extent = np.array([tsne_feature[0] - wp, tsne_feature[0] + wp,
                               tsne_feature[1] - hp, tsne_feature[1] + hp])
            ax.imshow(pos_im, extent=extent)

            extent = np.array([tsne_feature[0] - w, tsne_feature[0] + w,
                               tsne_feature[1] - h, tsne_feature[1] + h])
            ax.imshow(image, extent=extent, cmap=plt.cm.gray_r)
        
        # plot negative sim images
        for neg_sim_ind in neg_sim_indices:
            tsne_feature = tsne_features[neg_sim_ind,:]
            image = sub_sampled_images[neg_sim_ind,:,:,0]

            extent = np.array([tsne_feature[0] - w_real_sim, tsne_feature[0] + w_real_sim,
                               tsne_feature[1] - h_real_sim, tsne_feature[1] + h_real_sim])
            ax.imshow(sim_im, extent=extent)

            extent = np.array([tsne_feature[0] - wp, tsne_feature[0] + wp,
                               tsne_feature[1] - hp, tsne_feature[1] + hp])
            ax.imshow(neg_im, extent=extent)

            extent = np.array([tsne_feature[0] - w, tsne_feature[0] + w,
                               tsne_feature[1] - h, tsne_feature[1] + h])
            ax.imshow(image, extent=extent, cmap=plt.cm.gray_r)

        # plot positive sim images
        for pos_sim_ind in pos_sim_indices:
            tsne_feature = tsne_features[pos_sim_ind,:]
            image = sub_sampled_images[pos_sim_ind,:,:,0]

            extent = np.array([tsne_feature[0] - w_real_sim, tsne_feature[0] + w_real_sim,
                               tsne_feature[1] - h_real_sim, tsne_feature[1] + h_real_sim])
            ax.imshow(sim_im,  extent=extent)

            extent = np.array([tsne_feature[0] - wp, tsne_feature[0] + wp,
                               tsne_feature[1] - hp, tsne_feature[1] + hp])
            ax.imshow(neg_im,  extent=extent)

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

        vis2d.clf()
        # generate plots
        logging.info('Generating Standard T-SNE Plot of positives vs. negatives for layer {}'.format(layer))
        vis2d.figure(size=(figsize,figsize))
        vis2d.scatter(tsne_features[labels==0, 0], tsne_features[labels==0, 1], s=point_size, c='r', label='Failures')
        vis2d.scatter(tsne_features[labels==1, 0], tsne_features[labels==1, 1], s=point_size, c='b', label='Successes')
        vis2d.xlabel('T-SNE Dim 0', fontsize=font_size)
        vis2d.ylabel('T-SNE Dim 1', fontsize=font_size)
        vis2d.title('T-SNE for {}'.format(layer), fontsize=font_size)
        # vis2d.legend(loc='best')
        vis2d.savefig(os.path.join(output_dir, 'positives_vs._negatives_{}.pdf'.format(layer)), dpi=dpi)

        vis2d.clf()
        # generate plots
        logging.info('Generating Standard T-SNE Plot of real vs. sim for layer {}'.format(layer))
        vis2d.figure(size=(figsize,figsize))
        vis2d.scatter(tsne_features[real_sim_map==0, 0], tsne_features[real_sim_map==0, 1], s=point_size, c='g', label='Real')
        vis2d.scatter(tsne_features[real_sim_map==1, 0], tsne_features[real_sim_map==1, 1], s=point_size, c='y', label='Sim')
        vis2d.xlabel('T-SNE Dim 0', fontsize=font_size)
        vis2d.ylabel('T-SNE Dim 1', fontsize=font_size)
        vis2d.title('T-SNE for {}'.format(layer), fontsize=font_size)
        vis2d.legend(loc='best')
        vis2d.savefig(os.path.join(output_dir, 'real_vs._sim_{}.pdf'.format(layer)), dpi=dpi)