"""
Neon dataset wrapper for GQCNN datasets.
Author: Vishal Satish
"""
import os
import logging
import cPickle as pkl
import random

import numpy as np

from neon.data.datasets import Dataset

from gqcnn.utils.training_utils import setup_data_filenames, compute_indices_image_wise, \
    compute_indices_object_wise, compute_indices_pose_wise
from gqcnn.utils.data_utils import compute_data_metrics, compute_grasp_label_metrics
from gqcnn.utils.enums import InputPoseMode, InputGripperMode, DataSplitMode
from gqcnn_train_iterator import GQCNNTrainIterator
from gqcnn_val_iterator import GQCNNValIterator

class GQCNNDataset(Dataset):
    def __init__(self, gqcnn, experiment_dir, total_pct, train_pct, data_split_mode, target_metric_name, metric_thresh, data_dir, queue_sleep, queue_capacity,
        image_mode, training_mode, preproc_mode, cfg, debug=False, debug_num_files=100):
        self.experiment_dir = experiment_dir
        # create python lambda function to help create file paths to experiment_dir
        self.exp_path_gen = lambda fname: os.path.join(self.experiment_dir, fname)

        self.gqcnn = gqcnn
        self.cfg = cfg
        self.total_pct = total_pct
        self.train_pct = train_pct
        self.data_split_mode = data_split_mode
        self.target_metric_name = target_metric_name
        self.metric_thresh = metric_thresh
        self.data_dir = data_dir
        self.queue_sleep = queue_sleep
        self.queue_capacity = queue_capacity
        self.training_mode = training_mode
        self.preproc_mode = preproc_mode
        self.image_mode = image_mode
        self.debug = debug
        self.debug_num_files = debug_num_files
        self._setup()  

    @property
    def num_datapoints(self):
        return self._num_datapoints  

    def gen_iterators(self):
        self._data_iter_dict = {'train': GQCNNTrainIterator(self.im_filenames, self.pose_filenames, self.label_filenames, 
            self.pose_dim, self.input_pose_mode, self.im_width, self.im_height, self.im_channels, self.metric_thresh, self.data_dir, 
            self.queue_sleep, self.queue_capacity, self.training_mode, self.preproc_mode, self.train_index_map, self.im_mean, self.im_std, self.pose_mean, 
            self.pose_std, self.denoising_params, name='train_data')}
        self._data_iter_dict['val'] = GQCNNValIterator(self.im_filenames, self.pose_filenames, self.label_filenames, self.pose_dim, self.input_pose_mode,
            self.im_width, self.im_height, self.im_channels, self.metric_thresh, self.data_dir, self.training_mode, self.preproc_mode, self.val_index_map, 
            self.im_mean, self.im_std, self.pose_mean, self.pose_std, name='val_data')
        return self._data_iter_dict

    def _save_index_maps(self, train_idx_map, val_idx_map, train_fname, val_fname):
        with open(self.exp_path_gen(train_fname), 'w') as fhandle:
            pkl.dump(train_idx_map, fhandle)
        with open(self.exp_path_gen(val_fname), 'w') as fhandle:
            pkl.dump(val_idx_map, fhandle)

    def _compute_indices(self, data_split_mode, *computation_args):
        train_idx_map_fname = 'train_indices_{}.pkl'.format(data_split_mode)
        val_idx_map_fname = 'val_indices_{}.pkl'.format(data_split_mode)
        train_idx_map_fpath = self.exp_path_gen(train_idx_map_fname)
        val_idx_map_fpath = self.exp_path_gen(val_idx_map_fname)
        if os.path.exists(train_idx_map_fpath):
            with open(train_idx_map_fpath, 'r') as fhandle:
                train_idx_map = pkl.load(fhandle)
            with open(val_idx_map_fname, 'r') as fhandle:
                val_idx_map = pkl.load(fhandle)
        elif self.cfg['use_existing_indices']:
            with open(os.path.join(self.cfg['index_dir'], train_idx_map_fname)) as fhandle:
                train_idx_map = pkl.load(fhandle)
            with open(os.path.join(self.cfg['index_dir'], val_idx_map_fname)) as fhandle:
                val_idx_map = pkl.load(fhandle)
        else:
            if data_split_mode == DataSplitMode.IMAGE_WISE:
                train_idx_map, val_idx_map = compute_indices_image_wise(*computation_args)
            elif data_split_mode == DataSplitMode.OBJECT_WISE:
                train_idx_map, val_idx_map = compute_indices_object_wise(*computation_args)
            else:
                train_idx_map, val_idx_map = compute_indices_pose_wise(*computation_args)

        # save indices
        self._save_index_maps(train_idx_map, val_idx_map, train_idx_map_fname, val_idx_map_fname)

        return train_idx_map, val_idx_map

    def _compute_data_metrics(self):
        if self.cfg['fine_tune']:
            self.im_mean = self.gqcnn.get_im_mean()
            self.im_std = self.gqcnn.get_im_std()
            self.pose_mean = self.gqcnn.get_pose_mean()
            self.pose_std = self.gqcnn.get_pose_std()
            if self.gripper_dim > 0:
                self.gripper_mean = self.gqcnn.get_gripper_mean()
                self.gripper_std = self.gqcnn.get_gripper_std()
            elif self.input_gripper_mode == InputGripperMode.DEPTH_MASK:
                self.gripper_depth_mask_mean = self.gqcnn.get_gripper_depth_mask_mean()
                self.gripper_depth_mask_std = self.gqcnn.get_gripper_depth_mask_std()
        else:
            im_mean_fname = self.exp_path_gen('im_mean.npy')
            im_std_fname = self.exp_path_gen('im_std.npy')
            pose_mean_fname = self.exp_path_gen('pose_mean.npy')
            pose_std_fname = self.exp_path_gen('pose_std.npy')
            if self.gripper_dim > 0:
                gripper_mean_fname = self.exp_path_gen('gripper_mean.npy')
                gripper_std_fname = self.exp_path_gen('gripper_std.npy')
                self.image_mean, self.image_std, self.pose_mean, self.pose_std, self.gripper_mean, self.gripper_std, _, _ = compute_data_metrics(
                    self.experiment_dir, self.data_dir, self.im_height, self.im_width, self.pose_shape, self.input_pose_mode, self.train_index_map, 
                    self.im_filenames, self.pose_filenames, gripper_param_filenames=self.gripper_param_filenames, total_gripper_param_elems=self.gripper_shape, 
                    num_random_files=self.num_random_files)

                np.save(gripper_mean_fname, self.gripper_mean)
                np.save(gripper_std_fname, self.gripper_std)

                self.gqcnn.update_gripper_mean(self.gripper_mean)
                self.gqcnn.update_gripper_std(self.gripper_std)

            elif self.input_gripper_mode == InputGripperMode.DEPTH_MASK:
                gripper_depth_mask_mean_fname = self.exp_path_gen('gripper_depth_mask_mean.npy')
                gripper_depth_mask_std_fname = self.exp_path_gen('gripper_depth_mask_std.npy')
                self.image_mean, self.image_std, self.pose_mean, self.pose_std, _, _, self.gripper_depth_mask_mean, self.gripper_depth_mask_std = compute_data_metrics(
                    self.experiment_dir, self.data_dir, self.im_height, self.im_width, self.pose_shape, self.input_pose_mode, self.train_index_map, 
                    self.im_filenames, self.pose_filenames, gripper_depth_mask_filenames=self.gripper_depth_mask_filenames, num_random_files=self.num_random_files)

                np.save(gripper_depth_mask_mean_fname, self.gripper_depth_mask_mean)
                np.save(gripper_depth_mask_std_fname, self.gripper_depth_mask_std)

                self.gqcnn.update_gripper_depth_mask_mean(self.gripper_depth_mask_mean)
                self.gqcnn.update_gripper_depth_mask_std(self.gripper_depth_mask_std)

            else:
                self.im_mean, self.im_std, self.pose_mean, self.pose_std, _, _, _, _ = compute_data_metrics(
                    self.experiment_dir, self.data_dir, self.im_height, self.im_width, self.pose_shape, self.input_pose_mode, self.train_index_map, 
                    self.im_filenames, self.pose_filenames, num_random_files=self.num_random_files)

            np.save(im_mean_fname, self.im_mean)
            np.save(im_std_fname, self.im_std)
            np.save(pose_mean_fname, self.pose_mean)
            np.save(pose_std_fname, self.pose_std)

            self.gqcnn.update_im_mean(self.im_mean)
            self.gqcnn.update_im_std(self.im_std)
            self.gqcnn.update_pose_mean(self.pose_mean)
            self.gqcnn.update_pose_std(self.pose_std)

    def _setup(self):
        """ Setup Dataset """

        # read dataset filenames
        self.im_filenames, self.pose_filenames, self.label_filenames, self.gripper_param_filenames, \
        self.gripper_depth_mask_filenames, self.gripper_seg_mask_filenames,self.im_filenames_copy, \
        self.pose_filenames_copy, self.label_filenames_copy, self.gripper_param_filenames_copy, \
        self.gripper_depth_mask_filenames_copy, self.gripper_seg_mask_filenames_copy, self.obj_id_filenames, \
        self.stable_pose_filenames, self.num_files = setup_data_filenames(self.data_dir, self.image_mode, self.target_metric_name, self.total_pct, self.debug, self.debug_num_files)

        # read data parameters from config file
        self._read_data_params()

        # compute total number of datapoints in dataset(rounded up to num_datapoints_per_file)
        self._num_datapoints = self.images_per_file * self.num_files

        # compute train/test indices based on how the data is to be split
        if self.data_split_mode == DataSplitMode.IMAGE_WISE:
            self.train_index_map, self.val_index_map = self._compute_indices(DataSplitMode.IMAGE_WISE, self.data_dir, self.images_per_file, self._num_datapoints, self.train_pct, self.im_filenames)
        elif self.data_split_mode == DataSplitMode.OBJECT_WISE:
            self.train_index_map, self.val_index_map = self._compute_indices(DataSplitMode.OBJECT_WISE, self.data_dir, self.train_pct, self.im_filenames, self.obj_id_filenames)
        elif self.data_split_mode == DataSplitMode.STABLE_POSE_WISE:
            self.train_index_map, self.val_index_map = self._compute_indices(DataSplitMode.STABLE_POSE_WISE, self.data_dir, self.train_pct, self.im_filenames, self.stable_pose_filenames)
        else:
            raise ValueError('Data split mode: {} not supported'.format(self.data_split_mode))

        # compute data metrics
        self._compute_data_metrics()

        # compute grasp label metrics
        self.min_grasp_metric, self.max_grasp_metric, self.mean_grasp_metric, self.median_grasp_metric, pct_pos_val = compute_grasp_label_metrics(
            self.data_dir, self.im_filenames, self.label_filenames, self.val_index_map, self.metric_thresh)
        logging.info('Percent positive in val set: ' + str(pct_pos_val))

    def _read_data_params(self):
        """ Read data parameters from configuration file """

        self.train_im_data = np.load(os.path.join(self.data_dir, self.im_filenames[0]))['arr_0']
        self.pose_data = np.load(os.path.join(self.data_dir, self.pose_filenames[0]))['arr_0']
        self.metric_data = np.load(os.path.join(self.data_dir, self.label_filenames[0]))['arr_0']
        self.images_per_file = self.train_im_data.shape[0]
        self.im_height = self.train_im_data.shape[1]
        self.im_width = self.train_im_data.shape[2]
        self.im_channels = self.train_im_data.shape[3]

        self.num_tensor_channels = self.cfg['num_tensor_channels']
        self.pose_shape = self.pose_data.shape[1]
        self.input_pose_mode = self.cfg['input_pose_mode']
        self.input_gripper_mode = self.cfg['input_gripper_mode']

        # update pose dimension according to input_pose_mode for creation of tensorflow placeholders
        if self.input_pose_mode == InputPoseMode.TF_IMAGE:
            self.pose_dim = 1  # depth
        elif self.input_pose_mode == InputPoseMode.TF_IMAGE_PERSPECTIVE:
            self.pose_dim = 3  # depth, cx, cy
        elif self.input_pose_mode == InputPoseMode.RAW_IMAGE:
            self.pose_dim = 4  # u, v, theta, depth
        elif self.input_pose_mode == InputPoseMode.RAW_IMAGE_PERSPECTIVE:
            self.pose_dim = 6  # u, v, theta, depth cx, cy
        elif self.input_pose_mode == InputPoseMode.TF_IMAGE_SUCTION:
            self.pose_dim = 2  # depth, theta
        else:
            raise ValueError('Input pose mode: {} not understood'.format(self.input_pose_mode))

        # update gripper dimension according to input_gripper_mode for creation of tensorflow placeholders
        if self.input_gripper_mode == InputGripperMode.WIDTH:
            self.gripper_dim = 1  # width
        elif self.input_gripper_mode == InputGripperMode.NONE:
            self.gripper_dim = 0  # no gripper channel
        elif self.input_gripper_mode == InputGripperMode.ALL:
            self.gripper_dim = 4  # width, palm depth, fx, fy
        elif self.input_gripper_mode == InputGripperMode.DEPTH_MASK:
            self.gripper_dim = 0  # no gripper channel
            self.num_tensor_channels += 2  # masks will be added as channels to depth image
        else:
            raise ValueError('Input gripper mode: {} not understood'.format(self.input_gripper_mode))
        
        if self.gripper_dim > 0:
            self.gripper_data = np.load(os.path.join(self.data_dir, self.gripper_param_filenames[0]))['arr_0']
            self.gripper_shape = self.gripper_data.shape[1]        

        self.num_random_files = min(self.num_files, self.cfg['num_random_files'])

        self.denoising_params = self.cfg['denoise']
