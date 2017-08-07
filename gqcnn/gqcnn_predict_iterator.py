"""
DataIterator for predicting using GQ-CNN's. Supports multiple data inputs unlike standard
Neon ArrayIterator.
Author: Vishal Satish
"""

import numpy as np
import cv2

from neon.data import NervanaDataIterator

class GQCNNPredictIterator(NervanaDataIterator):
	def __init__(self, im_data, pose_data, lshape=None, name=None):
		"""
		During initialization, the input data will be converted to backend tensor objects
		(e.g. CPUTensor or GPUTensor). If the backend uses the GPU, the data is copied over to the
		device.
		Args:
			X (ndarray, shape: [# examples, feature size]): Input features of the
				dataset.
			y (ndarray, shape:[# examples, 1 or feature size], optional): Labels corresponding to
				the input features. If absent, the input features themselves will be returned as
				target values (e.g. autoencoder)
			nclass (int, optional): The number of classes in labels. Not necessary if
				labels are not provided or where the labels are non-categorical.
			lshape (tuple, optional): Local shape for the input features
				(e.g. # channels, height, width)
			make_onehot (bool, optional): True if y is a categorical label that has to be converted
				to a one hot representation.
		"""
		# Treat singletons like list so that iteration follows same syntax
		super(GQCNNPredictIterator, self).__init__(name=name)
		im_data = im_data if isinstance(im_data, list) else [im_data]
		pose_data = pose_data if isinstance(pose_data, list) else [pose_data]
		self.ndata = len(im_data[0])

		# pad input data if size along axis=0 is less than batch_size
		if self.ndata < self.be.bsz:
			padded_im_data = []
			padded_pose_data = []
			for im_arr, pose_arr in zip(im_data, pose_data):
				padded_im_arr = np.zeros((self.be.bsz, im_arr.shape[1]))
				padded_pose_arr = np.zeros((self.be.bsz, pose_arr.shape[1]))
				padded_im_arr[:self.ndata] = im_arr
				padded_pose_arr[:self.ndata] = pose_arr
				padded_im_data.append(padded_im_arr)
				padded_pose_data.append(padded_pose_arr)
			im_data = padded_im_data
			pose_data = padded_pose_data
			self.ndata = len(im_data[0])

		self.start = 0
		self.label_buf = None

		# make sure there are the same number of images and poses
		assert im_data[0].shape[0] == pose_data[0].shape[0], "Must have same number of images as poses"
		
		# if local shape is provided, then the product of lshape should match the
		# number of features
		if lshape is not None:
			lshape_im = lshape[0]
			lshape_pose = lshape[1]
			assert all([im_arr.shape[1] == np.prod(lshape_im) for im_arr in im_data]) and all([pose_arr.shape[1] == np.prod(lshape_pose) for pose_arr in pose_data]), \
				"product of lshape {} does not match input feature size".format(lshape)

		self.shape = lshape

		# Helpers to make dataset, minibatch, unpacking function for transpose and onehot
		def transpose_gen(z, x):
			return (self.be.array(z), self.be.array(x), self.be.iobuf(z.shape[1]), self.be.iobuf(x.shape[1]),
					lambda _in, _out: self.be.copy_transpose(_in, _out))

		self.im_dev, self.pose_dev, self.im_buf, self.pose_buf, self.unpack_func = list(zip(*[transpose_gen(im_arr, pose_arr) for im_arr, pose_arr in zip(im_data, pose_data)]))

		# Shallow copies for appending, iterating
		self.im_dbuf, self.im_hbuf, self.pose_dbuf, self.pose_hbuf = list(self.im_dev), list(self.im_buf), list(self.pose_dev), list(self.pose_buf)
		self.unpack_func = list(self.unpack_func)

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
		for i1 in range(self.start, self.ndata, self.be.bsz):
			bsz = min(self.be.bsz, self.ndata - i1)
			islice1, oslice1 = slice(0, bsz), slice(i1, i1 + bsz)
			islice2, oslice2 = None, None
			if self.be.bsz > bsz:
				islice2, oslice2 = slice(bsz, None), slice(0, self.be.bsz - bsz)
				self.start = self.be.bsz - bsz

			for im_buf, im_dev, pose_buf, pose_dev, unpack_func in zip(self.im_hbuf, self.im_dbuf, self.pose_hbuf, self.pose_dbuf, self.unpack_func):
				unpack_func(im_dev[oslice1], im_buf[:, islice1])
				unpack_func(pose_dev[oslice1], pose_buf[:, islice1])
				if oslice2:
					unpack_func(im_dev[oslice2], im_buf[:, islice2])
					unpack_func(pose_dev[oslice2], pose_buf[:, islice2])

			input_im = self.im_buf[0] if len(self.im_buf) == 1 else self.im_buf
			input_pose = self.pose_buf[0] if len(self.pose_buf) == 1 else self.pose_buf
			input_data = (input_im, input_pose)
			yield input_data, input_data