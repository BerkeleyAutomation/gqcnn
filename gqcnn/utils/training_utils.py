"""
Various helper functions for GQCNN training such as computing training indices,
copying training configurations, setting up output directories, etc.
Author: Vishal Satish
"""
import collections
import os
import json
import sys
import shutil
import logging

import numpy as np

from autolab_core import utils
from enums import DataFileTemplates, ImageMode, OutputDirTemplates

def copy_config(experiment_dir, cfg):
    """ Copy entire configuration dict and GQCNN architecture dict to JSON files in experiment_dir. Also copy
    training script to experiment_dir. """

    # copy entire configuration dict
    out_config_filename = os.path.join(experiment_dir, 'config.json')
    tempOrderedDict = collections.OrderedDict()
    for key in cfg.keys():
        tempOrderedDict[key] = cfg[key]
    with open(out_config_filename, 'w') as outfile:
        json.dump(tempOrderedDict, outfile)

    # copy GQCNN architecure dict
    out_architecture_filename = os.path.join(experiment_dir, 'architecture.json')
    json.dump(cfg['gqcnn_config']['architecture'], open(out_architecture_filename, 'w'))
    
    # copy training script
    this_filename = sys.argv[0]
    out_train_filename = os.path.join(experiment_dir, 'training_script.py')
    shutil.copyfile(this_filename, out_train_filename)

def compute_indices_image_wise(data_dir, images_per_file, num_datapoints, train_pct, im_filenames):
    """ Compute train and validation indices based on an image-wise split of the data"""

    # get total number of training datapoints
    num_train = int(train_pct * num_datapoints)
    
    # get training and validation indices
    all_indices = np.arange(num_datapoints)
    np.random.shuffle(all_indices)
    train_indices = np.sort(all_indices[:num_train])
    val_indices = np.sort(all_indices[num_train:])

    # make a map of the train and test indices for each file
    logging.info('Computing indices image-wise')
    train_index_map = {}
    val_index_map = {}
    for i, im_filename in enumerate(im_filenames):
        lower = i * images_per_file
        upper = (i + 1) * images_per_file
        im_arr = np.load(os.path.join(data_dir, im_filename))['arr_0']
        train_index_map[im_filename] = train_indices[(train_indices >= lower) & (
            train_indices < upper) & (train_indices - lower < im_arr.shape[0])] - lower
        val_index_map[im_filename] = val_indices[(val_indices >= lower) & (
            val_indices < upper) & (val_indices - lower < im_arr.shape[0])] - lower

    return train_index_map, val_index_map

def compute_indices_object_wise(data_dir, train_pct, im_filenames, obj_id_filenames):
    """ Compute train and validation indices based on an object-wise split"""

    # get number of unique objects by taking last object id of last object id file
    obj_id_filenames.sort(key=lambda x: int(x[-9:-4]))
    last_file_object_ids = np.load(os.path.join(
        data_dir, obj_id_filenames[len(obj_id_filenames) - 1]))['arr_0']
    num_unique_objs = last_file_object_ids[len(last_file_object_ids) - 1]
    num_train_obj = int(train_pct * num_unique_objs)
    logging.debug('There are: ' + str(num_unique_objs) + 'unique objects in this dataset.')

    # get training and validation indices
    all_object_ids = np.arange(num_unique_objs + 1)
    np.random.shuffle(all_object_ids)
    train_object_ids = np.sort(all_object_ids[:num_train_obj])
    val_object_ids = np.sort(all_object_ids[num_train_obj:])

    # make a map of the train and test indices for each file
    logging.info('Computing indices object-wise')
    train_index_map = {}
    val_index_map = {}
    for im_filename in im_filenames:
        # open up the corresponding obj_id file
        obj_ids = np.load(os.path.join(
            data_dir, 'object_labels_' + im_filename[-9:-4] + '.npz'))['arr_0']

        train_indices = []
        val_indices = []
        # for each obj_id if it is in train_object_ids then add it to train_indices else add it to val_indices
        for i, obj_id in enumerate(obj_ids):
            if obj_id in train_object_ids:
                train_indices.append(i)
            else:
                val_indices.append(i)

        train_index_map[im_filename] = np.asarray(train_indices, dtype=np.intc)
        val_index_map[im_filename] = np.asarray(val_indices, dtype=np.intc)

    return train_index_map, val_index_map

def compute_indices_pose_wise(data_dir, train_pct, im_filenames, stable_pose_filenames):
    """ Compute train and validation indices based on an image-stable-pose-wise split"""

    # get number of unique stable poses by taking last stable pose id of last stable pose id file
    stable_pose_filenames.sort(key=lambda x: int(x[-9:-4]))
    last_file_pose_ids = np.load(os.path.join(
        data_dir, stable_pose_filenames[len(stable_pose_filenames) - 1]))['arr_0']
    num_unique_stable_poses = last_file_pose_ids[len(last_file_pose_ids) - 1]
    num_train_poses = int(train_pct * num_unique_stable_poses)
    logging.debug('There are: ' + str(num_unique_stable_poses) +
                  'unique stable poses in this dataset.')

    # get training and validation indices
    all_pose_ids = np.arange(num_unique_stable_poses + 1)
    np.random.shuffle(all_pose_ids)
    train_pose_ids = np.sort(all_pose_ids[:num_train_poses])
    val_pose_ids = np.sort(all_pose_ids[num_train_poses:])

    # make a map of the train and test indices for each file
    logging.info('Computing indices stable-pose-wise')
    train_index_map = {}
    val_index_map = {}
    for im_filename in im_filenames:
        # open up the corresponding pose_id file
        pose_ids = np.load(os.path.join(
            data_dir, 'pose_labels_' + im_filename[-9:-4] + '.npz'))['arr_0']

        train_indices = []
        val_indices = []
        # for each pose_id if it is in train_pose_ids then add it to train_indices else add it to val_indices
        for i, pose_id in enumerate(pose_ids):
            if pose_id in train_pose_ids:
                train_indices.append(i)
            else:
                val_indices.append(i)

        train_index_map[im_filename] = np.asarray(train_indices, dtype=np.intc)
        val_index_map[im_filename] = np.asarray(val_indices, dtype=np.intc)

    return train_index_map, val_index_map

def get_decay_step(train_pct, num_datapoints, decay_step_multiplier):
    num_train = int(train_pct * num_datapoints)
    return decay_step_multiplier * num_train

def setup_data_filenames(data_dir, image_mode, target_metric_name, total_pct, debug, debug_num_files):
    """ Setup data filenames, subsample files, check validity of filenames"""

    # read in filenames of training data(poses, images, labels, obj_id's, stable_poses, gripper_params)
    logging.info('Reading filenames')
    all_filenames = os.listdir(data_dir)
    if image_mode == ImageMode.BINARY:
        im_filenames = [f for f in all_filenames if f.find(
            DataFileTemplates.binary_im_tensor_template) > -1]
    elif image_mode == ImageMode.DEPTH:
        im_filenames = [f for f in all_filenames if f.find(
            DataFileTemplates.depth_im_tensor_template) > -1]
    elif image_mode == ImageMode.BINARY_TF:
        im_filenames = [f for f in all_filenames if f.find(
            DataFileTemplates.binary_im_tf_tensor_template) > -1]
    elif image_mode == ImageMode.COLOR_TF:
        im_filenames = [f for f in all_filenames if f.find(
            DataFileTemplates.color_im_tf_tensor_template) > -1]
    elif image_mode == ImageMode.GRAY_TF:
        im_filenames = [f for f in all_filenames if f.find(
            DataFileTemplates.gray_im_tf_tensor_template) > -1]
    elif image_mode == ImageMode.DEPTH_TF:
        im_filenames = [f for f in all_filenames if f.find(
            DataFileTemplates.depth_im_tf_tensor_template) > -1]
    elif image_mode == ImageMode.DEPTH_TF_TABLE:
        im_filenames = [f for f in all_filenames if f.find(
            DataFileTemplates.depth_im_tf_table_tensor_template) > -1]
    else:
        raise ValueError('Image mode %s not supported.' % (image_mode))

    pose_filenames = [f for f in all_filenames if f.find(DataFileTemplates.hand_poses_template) > -1]
    label_filenames = [f for f in all_filenames if f.find(target_metric_name) > -1]
    # since these are not required in the dataset, we fill them with FileTemplates.FILENAME_PLACEHOLDER just to prevent sorting exceptions down the line
    # however, if they do not exist then exceptions will be thrown if the user tries to use object_wise/pose_wise splits
    # or tries to input the gripper parameters to the network during training
    obj_id_filenames = [f if (f.find(DataFileTemplates.object_labels_template) > -1) else DataFileTemplates.filename_placeholder for f in all_filenames]
    obj_id_files_exist = True
    if obj_id_filenames[0] == DataFileTemplates.filename_placeholder:
        obj_id_files_exist = False
    stable_pose_filenames = [f if (f.find(DataFileTemplates.pose_labels_template) > -1)
                                        else DataFileTemplates.filename_placeholder for f in all_filenames]
    stable_pose_files_exist = True
    if stable_pose_filenames[0] == DataFileTemplates.filename_placeholder:
        stable_pose_files_exist = False
    gripper_param_filenames = [f if (f.find(
        DataFileTemplates.gripper_params_template) > -1) else DataFileTemplates.filename_placeholder for f in all_filenames]
    gripper_depth_mask_filenames = [f if (f.find(
        DataFileTemplates.gripper_depth_template) > -1) else DataFileTemplates.filename_placeholder for f in all_filenames]
    gripper_seg_mask_filenames = [f if (f.find(
        DataFileTemplates.gripper_segmask_template) > -1) else DataFileTemplates.filename_placeholder for f in all_filenames]

    if debug:
        # sort
        im_filenames.sort(key=lambda x: int(x[-9:-4]))
        pose_filenames.sort(key=lambda x: int(x[-9:-4]))
        label_filenames.sort(key=lambda x: int(x[-9:-4]))
        obj_id_filenames.sort(key=lambda x: int(x[-9:-4]))
        stable_pose_filenames.sort(key=lambda x: int(x[-9:-4]))
        gripper_param_filenames.sort(key=lambda x: int(x[-9:-4]))
        gripper_depth_mask_filenames.sort(key = lambda x: int(x[-9:-4]))
        gripper_seg_mask_filenames.sort(key = lambda x: int(x[-9:-4]))

        # pack, shuffle and sample
        zipped = zip(im_filenames, pose_filenames, label_filenames, obj_id_filenames, 
            stable_pose_filenames, gripper_param_filenames, gripper_depth_mask_filenames, gripper_seg_mask_filenames)

        random.shuffle(zipped)
        zipped = zipped[:debug_num_files]

        # unpack
        im_filenames, pose_filenames, label_filenames, obj_id_filenames, stable_pose_filenames, gripper_param_filenames, 
        gripper_depth_mask_filenames, gripper_seg_mask_filenames = zip(*zipped)

    im_filenames.sort(key = lambda x: int(x[-9:-4]))
    pose_filenames.sort(key = lambda x: int(x[-9:-4]))
    label_filenames.sort(key = lambda x: int(x[-9:-4]))
    obj_id_filenames.sort(key = lambda x: int(x[-9:-4]))
    stable_pose_filenames.sort(key = lambda x: int(x[-9:-4]))
    gripper_param_filenames.sort(key = lambda x: int(x[-9:-4]))
    gripper_depth_mask_filenames.sort(key = lambda x: int(x[-9:-4]))
    gripper_seg_mask_filenames.sort(key = lambda x: int(x[-9:-4]))

    # check valid filenames
    if len(im_filenames) == 0 or len(pose_filenames) == 0 or len(label_filenames) == 0:
        raise RuntimeError('One or more required training files(Images, Poses, Labels) in the dataset could not be found.')

    # subsample files based on total_pct of the dataset to use
    num_files = len(im_filenames)
    num_files_used = int(total_pct * num_files)
    filename_indices = np.random.choice(num_files, size=num_files_used, replace=False)
    filename_indices.sort()
    im_filenames = [im_filenames[k] for k in filename_indices]
    pose_filenames = [pose_filenames[k] for k in filename_indices]
    label_filenames = [label_filenames[k] for k in filename_indices]
    obj_id_filenames = [obj_id_filenames[k] for k in filename_indices]   
    stable_pose_filenames = [stable_pose_filenames[k] for k in filename_indices]
    gripper_param_filenames = [gripper_param_filenames[k] for k in filename_indices]
    gripper_depth_mask_filenames = [gripper_depth_mask_filenames[k] for k in filename_indices]
    gripper_seg_mask_filenames = [gripper_seg_mask_filenames[k] for k in filename_indices]

    # create copy of image, pose, gripper_param, and label filenames because original cannot be accessed by load and enqueue op in the case 
    # the error_rate_in_batches method is sorting the original
    im_filenames_copy = im_filenames[:]
    pose_filenames_copy = pose_filenames[:]
    label_filenames_copy = label_filenames[:]
    gripper_param_filenames_copy = gripper_param_filenames[:]
    gripper_depth_mask_filenames_copy = gripper_depth_mask_filenames[:]
    gripper_seg_mask_filenames_copy = gripper_seg_mask_filenames[:]

    return im_filenames, pose_filenames, label_filenames, gripper_param_filenames, gripper_depth_mask_filenames, gripper_seg_mask_filenames, \
        im_filenames_copy, pose_filenames_copy, label_filenames_copy, gripper_param_filenames_copy, gripper_depth_mask_filenames_copy, \
        gripper_seg_mask_filenames_copy, obj_id_filenames, stable_pose_filenames, num_files 

def setup_output_dirs(output_dir):
    """ Setup output directories """

    # setup experiment_dir
    experiment_id = utils.gen_experiment_id()
    experiment_dir = os.path.join(output_dir, OutputDirTemplates.MODEL_DIR + '_%s' %(experiment_id))
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    # setup summary_dir    
    summary_dir = os.path.join(experiment_dir, OutputDirTemplates.SUMMARY_DIR)
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    else:
        # if the summary directory already exists, clean it out by deleting all files in it,
        # we don't want tensorboard to get confused with old logs while debugging with the same directory
        old_files = os.listdir(summary_dir)
        for file in old_files:
            os.remove(os.path.join(summary_dir, file))

    logging.info('Saving model to %s' %(experiment_dir))

    # setup filter directory
    filter_dir = os.path.join(experiment_dir, OutputDirTemplates.FILTER_DIR)
    if not os.path.exists(filter_dir):
        os.mkdir(filter_dir)

    return experiment_dir, summary_dir, filter_dir

def setup_python_logger():
    logging.getLogger().setLevel(logging.INFO)
