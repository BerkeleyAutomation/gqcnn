"""
Custom NervanaDataIterator for feeding validation data to GQCNN network during training.
Author: Vishal Satish
"""
import os

import numpy as np

from neon.data import NervanaDataIterator

from gqcnn.utils.data_utils import parse_pose_data
from gqcnn.utils.enums import TrainingMode, PreprocMode

class GQCNNValIterator(NervanaDataIterator):
    def __init__(self, im_filenames, pose_filenames, label_filenames, pose_dim, input_pose_mode, im_width, 
        im_height, im_channels, metric_thresh, dataset_dir, training_mode, preproc_mode, 
        indices, im_mean, im_std, pose_mean, pose_std, make_onehot=True, nclass=2, name=None):
        super(GQCNNValIterator, self).__init__(name=name)

        self.make_onehot = make_onehot
        self.nclass = nclass
        self.im_filenames = im_filenames
        self.pose_filenames = pose_filenames
        self.label_filenames = label_filenames
        self.indices = indices
        self.dataset_dir = dataset_dir
        self.input_pose_mode = input_pose_mode
        self.metric_thresh = metric_thresh
        self.im_height = im_height
        self.im_width = im_width
        self.im_channels = im_channels
        self.training_mode = training_mode
        self.preproc_mode = preproc_mode
        self.im_mean = im_mean
        self.im_std = im_std
        self.pose_mean = pose_mean
        self.pose_std = pose_std
        self.pose_dim = pose_dim

        # sort filename arrays
        self.im_filenames.sort(key = lambda x: int(x[-9:-4]))
        self.pose_filenames.sort(key = lambda x: int(x[-9:-4]))
        self.label_filenames.sort(key = lambda x: int(x[-9:-4]))

        # set shape
        self.shape = [(self.im_channels, self.im_height, self.im_width), (self.pose_dim,)]

        # calculate number of datapoints
        self.ndata = 0
        for file in self.indices:
            self.ndata += len(self.indices[file])

        # setup label datatype
        if self.training_mode == TrainingMode.REGRESSION:
            self.numpy_dtype = np.float32
        elif self.training_mode == TrainingMode.CLASSIFICATION:
            self.numpy_dtype = np.int64

        # create transpose function for fast gpu transposes using neon backend
        self.transpose_func = lambda _in, _out: self.be.copy_transpose(_in, _out)

        # create one-hot function for neon backend one-hot
        self.onehot_func = lambda _in, _out: self.be.onehot(_in, axis=0, out=_out)

        self.start = 0

    @property
    def nbatches(self):
        """
        Return the number of minibatches in this dataset.
        """
        return -((self.start - self.ndata) // self.be.bsz)

    def reset(self):
        """
        Resets the starting index of this dataset to zero. Useful for calling
        repeated evaluations on the dataset without having to wrap around
        the last uneven minibatch. Not necessary when data is divisible by batch size
        """
        self.start = 0

    def __iter__(self):
        """
        Returns a new minibatch of data with each call.
        Yields:
            tuple: The next minibatch which includes both features and labels.
        """
        file_idx = 0
        start_ind_data = 0
        end_ind_data = 0
        for i in range(self.start, self.ndata, self.be.bsz):
            im_arr = None
            pose_arr = None
            label_arr = None
            start_ind_tensor = 0
            end_ind_tensor = 0
            while start_ind_tensor < self.be.bsz:
                indices = self.indices[self.im_filenames[file_idx]]

                file_im_data = np.load(os.path.join(self.dataset_dir, self.im_filenames[file_idx]))['arr_0'][indices]
                file_pose_data = parse_pose_data(np.load(os.path.join(self.dataset_dir, self.pose_filenames[file_idx]))['arr_0'][indices], self.input_pose_mode)
                file_label_data = np.load(os.path.join(self.dataset_dir, self.label_filenames[file_idx]))['arr_0'][indices]

                # allocate arrays
                if im_arr is None:
                    im_arr = np.zeros((self.be.bsz, ) + file_im_data.shape[1:])
                if pose_arr is None:
                    pose_arr = np.zeros((self.be.bsz, ) + file_pose_data.shape[1:])
                if label_arr is None:
                    label_arr = np.zeros((self.be.bsz, ) + file_label_data.shape[1:])

                # threshold or normalize labels
                if self.training_mode == TrainingMode.REGRESSION:
                    if self.preproc_mode == PreprocMode.NORMALIZATION:
                        file_label_data = (file_label_data - self.min_metric) / (self.max_metric - self.min_metric)
                elif self.training_mode == TrainingMode.CLASSIFICATION:
                    file_label_data = 1 * (file_label_data > self.metric_thresh)
                    file_label_data = file_label_data.astype(self.numpy_dtype)

                # add data to arrays
                end_ind_tensor = min(start_ind_tensor + file_im_data.shape[0] - start_ind_data, self.be.bsz)
                end_ind_data = start_ind_data + end_ind_tensor - start_ind_tensor

                im_arr[start_ind_tensor:end_ind_tensor] = file_im_data[start_ind_data:end_ind_data]
                pose_arr[start_ind_tensor:end_ind_tensor] = file_pose_data[start_ind_data:end_ind_data]
                label_arr[start_ind_tensor:end_ind_tensor] = file_label_data[start_ind_data:end_ind_data]
                
                start_ind_tensor = end_ind_tensor
                start_ind_data = end_ind_data
                if start_ind_data >= file_im_data.shape[0]:
                    start_ind_data = 0
                    file_idx = (file_idx + 1) % len(self.im_filenames)

            im_arr = (im_arr - self.im_mean) / self.im_std
            pose_arr = (pose_arr - parse_pose_data(self.pose_mean, self.input_pose_mode)) / parse_pose_data(self.pose_std, self.input_pose_mode)
            
            im_arr_flat = im_arr.reshape((self.be.bsz, self.im_width * self.im_height * self.im_channels))

            # load the data into device memory and perform faster transpose using neon backend
            im_arr_dev = self.be.array(im_arr_flat)
            pose_arr_dev = self.be.array(pose_arr)

            if self.make_onehot:
                label_arr_dev = self.be.array(label_arr.reshape((1, -1)), dtype=np.int32)
            else:
                label_arr_dev = self.be.array(label_arr)

            im_arr_buf = self.be.iobuf(im_arr_flat.shape[1], dtype=np.float32)
            pose_arr_buf = self.be.iobuf(pose_arr.shape[1], dtype=np.float32)

            if self.make_onehot:
                label_arr_buf = self.be.iobuf(self.nclass)
            else:
                label_arr_buf = self.be.iobuf(label_arr.shape[1:])

            self.transpose_func(im_arr_dev, im_arr_buf)
            self.transpose_func(pose_arr_dev, pose_arr_buf)

            if self.make_onehot:
                self.onehot_func(label_arr_dev, label_arr_buf)
            else:
                self.transpose_func(label_arr_dev, label_arr_buf)
            
            yield (im_arr_buf, pose_arr_buf), label_arr_buf
