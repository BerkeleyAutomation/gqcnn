"""
Various helper functions.
Author: Vishal Satish
"""
import collections
import os
import json
import sys
import shutil

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

def compute_indices_image_wise(images_per_file, num_files):
    """ Compute train and validation indices based on an image-wise split of the data"""

    # get total number of training datapoints and set the decay_step
    num_datapoints = images_per_file * num_files
    
    # get training and validation indices
    all_indices = np.arange(num_datapoints)
    np.random.shuffle(all_indices)
    train_indices = np.sort(all_indices[:self.num_train])
    val_indices = np.sort(all_indices[self.num_train:])

    # make a map of the train and test indices for each file
    logging.info('Computing indices image-wise')
    train_index_map_filename = os.path.join(
        self.experiment_dir, 'train_indices_image_wise.pkl')
    self.val_index_map_filename = os.path.join(
        self.experiment_dir, 'val_indices_image_wise.pkl')
    if os.path.exists(train_index_map_filename):
        self.train_index_map = pkl.load(open(train_index_map_filename, 'r'))
        self.val_index_map = pkl.load(open(self.val_index_map_filename, 'r'))
    elif self.cfg['use_existing_indices']:
        self.train_index_map = pkl.load(open(os.path.join(self.cfg['index_dir'], 'train_indices_image_wise.pkl'), 'r'))
        self.val_index_map = pkl.load(open(os.path.join(self.cfg['index_dir'], 'val_indices_image_wise.pkl'), 'r'))
    else:
        self.train_index_map = {}
        self.val_index_map = {}
        for i, im_filename in enumerate(self.im_filenames):
            lower = i * self.images_per_file
            upper = (i + 1) * self.images_per_file
            im_arr = np.load(os.path.join(self.data_dir, im_filename))['arr_0']
            self.train_index_map[im_filename] = train_indices[(train_indices >= lower) & (
                train_indices < upper) & (train_indices - lower < im_arr.shape[0])] - lower
            self.val_index_map[im_filename] = val_indices[(val_indices >= lower) & (
                val_indices < upper) & (val_indices - lower < im_arr.shape[0])] - lower
        pkl.dump(self.train_index_map, open(train_index_map_filename, 'w'))
        pkl.dump(self.val_index_map, open(self.val_index_map_filename, 'w'))

def _compute_indices_object_wise(self):
    """ Compute train and validation indices based on an object-wise split"""

    # throw an excpetion if the object ids are not in the dataset
    if not self._obj_files_exist:
        raise RuntimeError('Object Id Files were not found in dataset')

    # get total number of training datapoints and set the decay_step
    num_datapoints = self.images_per_file * self.num_files
    self.num_train = int(self.train_pct * num_datapoints)
    self.decay_step = self.decay_step_multiplier * self.num_train

    # get number of unique objects by taking last object id of last object id file
    self.obj_id_filenames.sort(key=lambda x: int(x[-9:-4]))
    last_file_object_ids = np.load(os.path.join(
        self.data_dir, self.obj_id_filenames[len(self.obj_id_filenames) - 1]))['arr_0']
    num_unique_objs = last_file_object_ids[len(last_file_object_ids) - 1]
    self.num_train_obj = int(self.train_pct * num_unique_objs)
    logging.debug('There are: ' + str(num_unique_objs) +
                  'unique objects in this dataset.')

    # get training and validation indices
    all_object_ids = np.arange(num_unique_objs + 1)
    np.random.shuffle(all_object_ids)
    train_object_ids = np.sort(all_object_ids[:self.num_train_obj])
    val_object_ids = np.sort(all_object_ids[self.num_train_obj:])

    # make a map of the train and test indices for each file
    logging.info('Computing indices object-wise')
    train_index_map_filename = os.path.join(
        self.experiment_dir, 'train_indices_object_wise.pkl')
    self.val_index_map_filename = os.path.join(
        self.experiment_dir, 'val_indices_object_wise.pkl')
    if os.path.exists(train_index_map_filename):
        self.train_index_map = pkl.load(open(train_index_map_filename, 'r'))
        self.val_index_map = pkl.load(open(self.val_index_map_filename, 'r'))
    else:
        self.train_index_map = {}
        self.val_index_map = {}
        for im_filename in self.im_filenames:
            # open up the corresponding obj_id file
            obj_ids = np.load(os.path.join(
                self.data_dir, 'object_labels_' + im_filename[-9:-4] + '.npz'))['arr_0']

            train_indices = []
            val_indices = []
            # for each obj_id if it is in train_object_ids then add it to train_indices else add it to val_indices
            for i, obj_id in enumerate(obj_ids):
                if obj_id in train_object_ids:
                    train_indices.append(i)
                else:
                    val_indices.append(i)

            self.train_index_map[im_filename] = np.asarray(
                train_indices, dtype=np.intc)
            self.val_index_map[im_filename] = np.asarray(val_indices, dtype=np.intc)
            train_indices = []
            val_indices = []

        pkl.dump(self.train_index_map, open(train_index_map_filename, 'w'))
        pkl.dump(self.val_index_map, open(self.val_index_map_filename, 'w'))

def _compute_indices_pose_wise(self):
    """ Compute train and validation indices based on an image-stable-pose-wise split"""

    # throw an excpetion if the stable_pose_labels are not in the dataset
    if not self._stable_pose_files_exist:
        raise RuntimeError('Stable Pose Files were not found in dataset')

    # get total number of training datapoints and set the decay_step
    num_datapoints = self.images_per_file * self.num_files
    self.num_train = int(self.train_pct * num_datapoints)
    self.decay_step = self.decay_step_multiplier * self.num_train

    # get number of unique stable poses by taking last stable pose id of last stable pose id file
    self.stable_pose_filenames.sort(key=lambda x: int(x[-9:-4]))
    last_file_pose_ids = np.load(os.path.join(
        self.data_dir, self.stable_pose_filenames[len(self.stable_pose_filenames) - 1]))['arr_0']
    num_unique_stable_poses = last_file_pose_ids[len(last_file_pose_ids) - 1]
    self.num_train_poses = int(self.train_pct * num_unique_stable_poses)
    logging.debug('There are: ' + str(num_unique_stable_poses) +
                  'unique stable poses in this dataset.')

    # get training and validation indices
    all_pose_ids = np.arange(num_unique_stable_poses + 1)
    np.random.shuffle(all_pose_ids)
    train_pose_ids = np.sort(all_pose_ids[:self.num_train_poses])
    val_pose_ids = np.sort(all_pose_ids[self.num_train_poses:])

    # make a map of the train and test indices for each file
    logging.info('Computing indices stable-pose-wise')
    train_index_map_filename = os.path.join(
        self.experiment_dir, 'train_indices_stable_pose_wise.pkl')
    self.val_index_map_filename = os.path.join(
        self.experiment_dir, 'val_indices_stable_pose_wise.pkl')
    if os.path.exists(train_index_map_filename):
        self.train_index_map = pkl.load(open(train_index_map_filename, 'r'))
        self.val_index_map = pkl.load(open(self.val_index_map_filename, 'r'))
    else:
        self.train_index_map = {}
        self.val_index_map = {}
        for im_filename in self.im_filenames:
            # open up the corresponding obj_id file
            pose_ids = np.load(os.path.join(
                self.data_dir, 'pose_labels_' + im_filename[-9:-4] + '.npz'))['arr_0']

            train_indices = []
            val_indices = []
            # for each obj_id if it is in train_object_ids then add it to train_indices else add it to val_indices
            for i, pose_id in enumerate(pose_ids):
                if pose_id in train_pose_ids:
                    train_indices.append(i)
                else:
                    val_indices.append(i)

            self.train_index_map[im_filename] = np.asarray(
                train_indices, dtype=np.intc)
            self.val_index_map[im_filename] = np.asarray(val_indices, dtype=np.intc)
            train_indices = []
            val_indices = []

        pkl.dump(self.train_index_map, open(train_index_map_filename, 'w'))
        pkl.dump(self.val_index_map, open(self.val_index_map_filename, 'w'))

def get_decay_step(train_pct, num_datapoints, decay_step_multiplier):
	num_train = int(train_pct * num_datapoints)
    return decay_step_multiplier * num_train

