"""
Custom NervanaDataIterator for feeding training data to GQCNN network during training.
Author: Vishal Satish
"""
import os
import sys
import Queue
import threading
import time

import numpy as np

from neon.data import NervanaDataIterator

from gqcnn.utils.data_utils import denoise, parse_pose_data
from gqcnn.utils.enums import TrainingMode, PreprocMode

class GQCNNTrainIterator(NervanaDataIterator):
    def __init__(self, im_filenames, pose_filenames, label_filenames, pose_dim, input_pose_mode, im_width, 
        im_height, im_channels, metric_thresh, dataset_dir, queue_sleep, queue_capacity, 
        training_mode, preproc_mode, indices, im_mean, im_std, pose_mean, pose_std, 
        denoising_params, make_onehot=True, nclass=2, name=None):
        super(GQCNNTrainIterator, self).__init__(name=name)

        self.make_onehot = make_onehot
        self.nclass = nclass
        self.im_filenames = im_filenames
        self.pose_filenames = pose_filenames
        self.label_filenames = label_filenames
        self.indices = indices
        self.queue_capacity = queue_capacity
        self.queue_sleep = queue_sleep
        self.dataset_dir = dataset_dir
        self.metric_thresh = metric_thresh
        self.im_height = im_height
        self.im_width = im_width
        self.im_channels = im_channels
        self.training_mode = training_mode
        self.preproc_mode = preproc_mode
        self.input_pose_mode = input_pose_mode
        self.im_center = np.array([float(self.im_height-1)/2, float(self.im_width-1)/2])
        self.im_mean = im_mean
        self.im_std = im_std
        self.pose_mean = pose_mean
        self.pose_std = pose_std
        self.pose_dim = pose_dim
        self.denoising_params = denoising_params

        # sort filename arrays
        self.im_filenames.sort(key = lambda x: int(x[-9:-4]))
        self.pose_filenames.sort(key = lambda x: int(x[-9:-4]))
        self.label_filenames.sort(key = lambda x: int(x[-9:-4]))

        # set shape
        self.shape = [(self.im_channels, self.im_height, self.im_width), (self.pose_dim, )]

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
    
        self.queue = Queue.Queue(maxsize=self.queue_capacity)
        self.term_event = threading.Event()
        self.queue_thread = threading.Thread(target=self._load_and_enqueue)
        self.queue_thread.start()
        while self.queue.qsize() < self.queue_capacity:
            time.sleep(.001)

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

    def set_term_event(self):
        self.term_event.set()

    def _load_and_enqueue(self):
        while not self.term_event.is_set():
            time.sleep(.001)

            im_arr = None
            pose_arr = None
            label_arr = None
            start_ind = 0
            while start_ind < self.be.bsz:
                index = np.random.choice(len(self.im_filenames))
                indices = self.indices[self.im_filenames[index]]
                np.random.shuffle(indices)

                file_im_data = np.load(os.path.join(self.dataset_dir, self.im_filenames[index]))['arr_0'][indices]
                file_pose_data = parse_pose_data(np.load(os.path.join(self.dataset_dir, self.pose_filenames[index]))['arr_0'][indices], self.input_pose_mode)
                file_label_data = np.load(os.path.join(self.dataset_dir, self.label_filenames[index]))['arr_0'][indices]

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

                # distort(add noise) to images
                file_im_data, file_pose_data = denoise(file_im_data, self.im_height, self.im_width, self.im_channels, self.denoising_params, pose_arr=file_pose_data, pose_dim=self.pose_dim)

                # add data to arrays
                if file_im_data.shape[0] > self.be.bsz:
                    end_ind = self.be.bsz
                else:
                    end_ind = min(start_ind + file_im_data.shape[0], self.be.bsz)

                im_arr[start_ind:end_ind] = file_im_data[:end_ind - start_ind]
                pose_arr[start_ind:end_ind] = file_pose_data[:end_ind - start_ind]
                label_arr[start_ind:end_ind] = file_label_data[:end_ind - start_ind]

                # update start index
                start_ind = end_ind

            im_arr = (im_arr - self.im_mean) / self.im_std
            pose_arr = (pose_arr - self.pose_mean[2:3]) / self.pose_std[2:3]

            if not self.term_event.is_set():
                self.queue.put((im_arr, pose_arr, label_arr))

    def __iter__(self):
        """
        Returns a new minibatch of data with each call.
        Yields:
            tuple: The next minibatch which includes both features and labels.
        """
        for i in range(self.start, self.ndata, self.be.bsz):
            im_arr, pose_arr, label_arr = self.queue.get()
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
