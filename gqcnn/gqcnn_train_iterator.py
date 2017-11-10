"""
NervanaDataIterator for feeding training data to GQ-CNN network for training.
 
Author: Vishal Satish
"""
import os
import cv2
import numpy as np
import scipy.misc as sm
import scipy.ndimage.filters as sf
import scipy.ndimage.morphology as snm
import scipy.stats as ss
import skimage.draw as sd
import IPython

from neon.data import NervanaDataIterator

from gqcnn import TrainingMode, InputDataMode, PreprocMode

class GQCNNTrainIterator(NervanaDataIterator):
	def __init__(self, im_filenames, pose_filenames, label_filenames, indices, train_config, im_mean, im_std, pose_mean, pose_std, training_mode=TrainingMode.CLASSIFICATION, 
		preproc_mode=PreprocMode.NORMALIZATION, distort=False, make_onehot=True, nclass=2, name=None):
		
        # Treat singletons like list so that iteration follows same syntax
		super(GQCNNTrainIterator, self).__init__(name=name)

		#####################################READ PARAMETERS######################################## 
		self.distort = distort
		self.make_onehot = make_onehot
		self.nclass = nclass
		self.training_mode = training_mode
		self.preproc_mode = preproc_mode
		self.im_filenames = im_filenames
		self.pose_filenames = pose_filenames
		self.label_filenames = label_filenames
		self.indices = indices
		self.cfg = train_config
		self.dataset_dir = self.cfg['dataset_dir']
		self.input_data_mode = self.cfg['input_data_mode']
		self.metric_thresh = self.cfg['metric_thresh']
		self.im_height = self.cfg['gqcnn_config']['im_height']
		self.im_width = self.cfg['gqcnn_config']['im_width']
		self.im_channels = self.cfg['gqcnn_config']['im_channels']
		self.im_center = np.array([float(self.im_height-1)/2, float(self.im_width-1)/2])
		self.im_mean = im_mean
		self.im_std = im_std
		self.pose_mean = pose_mean
		self.pose_std = pose_std

		if self.input_data_mode == InputDataMode.TF_IMAGE:
			self.pose_dim = 1 # depth
		elif self.input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
			self.pose_dim = 3 # depth, cx, cy
		elif self.input_data_mode == InputDataMode.RAW_IMAGE:
			self.pose_dim = 4 # u, v, theta, depth
		elif self.input_data_mode == InputDataMode.RAW_IMAGE_PERSPECTIVE:
			self.pose_dim = 6 # u, v, theta, depth cx, cy
		else:
			raise ValueError('Input data mode %s not understood' %(self.input_data_mode))

		if self.distort:
			# read parameters of gaussian process for distortion
			self.gp_rescale_factor = self.cfg['gaussian_process_scaling_factor']
			self.gp_sample_height = int(self.im_height / self.gp_rescale_factor)
			self.gp_sample_width = int(self.im_width / self.gp_rescale_factor)
			self.gp_num_pix = self.gp_sample_height * self.gp_sample_width
			self.gp_sigma = self.cfg['gaussian_process_sigma']
			self.gamma_shape = self.cfg['gamma_shape']
			self.gamma_scale = 1.0 / self.gamma_shape

		# sort filename arrays
		self.im_filenames.sort(key = lambda x: int(x[-9:-4]))
		self.pose_filenames.sort(key = lambda x: int(x[-9:-4]))
		self.label_filenames.sort(key = lambda x: int(x[-9:-4]))

		# set shape
		self.shape = [(self.im_channels, self.im_height, self.im_width), (self.pose_dim, )]
	    #     self.shape = (self.im_channels, self.im_height, self.im_width)

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

	def _read_pose_data(self, pose_arr, input_data_mode):
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
		else:
			raise ValueError('Input data mode %s not supported. The RAW_* input data modes have been deprecated.' %(input_data_mode))

	def _distort(self, im_arr, pose_arr):
		""" Adds noise to a batch of images and poses"""
		# denoising and synthetic data generation
		if self.cfg['multiplicative_denoising']:
			mult_samples = ss.gamma.rvs(self.gamma_shape, scale=self.gamma_scale, size=len(im_arr))
			mult_samples = mult_samples[:,np.newaxis,np.newaxis,np.newaxis]
			im_arr = im_arr * np.tile(mult_samples, [1, self.im_height, self.im_width, self.im_channels])

		# randomly dropout regions of the image for robustness
		if self.cfg['image_dropout']:
			for i in range(len(im_arr)):
				if np.random.rand() < self.cfg['image_dropout_rate']:
					image = self.im_arr[i,:,:,0]
					nonzero_px = np.where(image > 0)
					nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]]
					num_nonzero = nonzero_px.shape[0]
					num_dropout_regions = ss.poisson.rvs(self.cfg['dropout_poisson_mean']) 
					
					# sample ellipses
					dropout_centers = np.random.choice(num_nonzero, size=num_dropout_regions)
					x_radii = ss.gamma.rvs(self.cfg['dropout_radius_shape'], scale=self.cfg['dropout_radius_scale'], size=num_dropout_regions)
					y_radii = ss.gamma.rvs(self.cfg['dropout_radius_shape'], scale=self.cfg['dropout_radius_scale'], size=num_dropout_regions)

					# set interior pixels to zero
					for j in range(num_dropout_regions):
						ind = dropout_centers[j]
						dropout_center = nonzero_px[ind, :]
						x_radius = x_radii[j]
						y_radius = y_radii[j]
						dropout_px_y, dropout_px_x = sd.ellipse(dropout_center[0], dropout_center[1], y_radius, x_radius, shape=image.shape)
						image[dropout_px_y, dropout_px_x] = 0.0
					im_arr[i,:,:,0] = image

		# dropout a region around the areas of the image with high gradient
		if self.cfg['gradient_dropout']:
			for i in range(len(im_arr)):
				if np.random.rand() < self.cfg['gradient_dropout_rate']:
					image = im_arr[i,:,:,0]
					grad_mag = sf.gaussian_gradient_magnitude(image, sigma=self.cfg['gradient_dropout_sigma'])
					thresh = ss.gamma.rvs(self.cfg['gradient_dropout_shape'], self.cfg['gradient_dropout_scale'], size=1)
					high_gradient_px = np.where(grad_mag > thresh)
					image[high_gradient_px[0], high_gradient_px[1]] = 0.0
				im_arr[i,:,:,0] = image

		# add correlated Gaussian noise
		if self.cfg['gaussian_process_denoising']:
			for i in range(len(im_arr)):
				if np.random.rand() < self.cfg['gaussian_process_rate']:
					image = im_arr[i,:,:,0]
					gp_noise = ss.norm.rvs(scale=self.gp_sigma, size=self.gp_num_pix).reshape(self.gp_sample_height, self.gp_sample_width)
					gp_noise = sm.imresize(gp_noise, self.gp_rescale_factor, interp='bicubic', mode='F')
					image[image > 0] += gp_noise[image > 0]
					im_arr[i,:,:,0] = image

		# run open and close filters to 
		if self.cfg['morphological']:
			for i in range(len(im_arr)):
				image = im_arr[i,:,:,0]
				sample = np.random.rand()
				morph_filter_dim = ss.poisson.rvs(self.cfg['morph_poisson_mean'])                         
				if sample < self.cfg['morph_open_rate']:
					image = snm.grey_opening(image, size=morph_filter_dim)
				else:
					closed_train_image = snm.grey_closing(image, size=morph_filter_dim)
					
					# set new closed pixels to the minimum depth, mimicing the table
					new_nonzero_px = np.where((image == 0) & (closed_train_image > 0))
					closed_train_image[new_nonzero_px[0], new_nonzero_px[1]] = np.min(image[image>0])
					image = closed_train_image.copy()

				im_arr[i,:,:,0] = image                        

		# randomly dropout borders of the image for robustness
		if self.cfg['border_distortion']:
			for i in range(len(im_arr)):
				image = im_arr[i,:,:,0]
				grad_mag = sf.gaussian_gradient_magnitude(image, sigma=self.cfg['border_grad_sigma'])
				high_gradient_px = np.where(grad_mag > self.cfg['border_grad_thresh'])
				high_gradient_px = np.c_[high_gradient_px[0], high_gradient_px[1]]
				num_nonzero = high_gradient_px.shape[0]
				num_dropout_regions = ss.poisson.rvs(self.cfg['border_poisson_mean']) 

				# sample ellipses
				dropout_centers = np.random.choice(num_nonzero, size=num_dropout_regions)
				x_radii = ss.gamma.rvs(self.cfg['border_radius_shape'], scale=self.cfg['border_radius_scale'], size=num_dropout_regions)
				y_radii = ss.gamma.rvs(self.cfg['border_radius_shape'], scale=self.cfg['border_radius_scale'], size=num_dropout_regions)

				# set interior pixels to zero or one
				for j in range(num_dropout_regions):
					ind = dropout_centers[j]
					dropout_center = high_gradient_px[ind, :]
					x_radius = x_radii[j]
					y_radius = y_radii[j]
					dropout_px_y, dropout_px_x = sd.ellipse(dropout_center[0], dropout_center[1], y_radius, x_radius, shape=image.shape)
					if np.random.rand() < 0.5:
						image[dropout_px_y, dropout_px_x] = 0.0
					else:
						image[dropout_px_y, dropout_px_x] = image[dropout_center[0], dropout_center[1]]

				im_arr[i,:,:,0] = image

		# randomly replace background pixels with constant depth
		if self.cfg['background_denoising']:
			for i in range(len(im_arr)):
				image = self.im_arr[i,:,:,0]                
				if np.random.rand() < self.cfg['background_rate']:
					image[image > 0] = self.cfg['background_min_depth'] + (self.cfg['background_max_depth'] - self.cfg['background_min_depth']) * np.random.rand()
				im_arr[i,:,:,0] = image

		# symmetrize images and poses
		if self.cfg['symmetrize']:
			for i in range(len(im_arr)):
				image = im_arr[i,:,:,0]
				# rotate with 50% probability
				if np.random.rand() < 0.5:
					theta = 180.0
					rot_map = cv2.getRotationMatrix2D(tuple(self.im_center), theta, 1)
					image = cv2.warpAffine(image, rot_map, (self.im_height, self.im_width), flags=cv2.INTER_NEAREST)
					if self.pose_dim > 1:
						pose_arr[i,4] = -pose_arr[i,4]
						pose_arr[i,5] = -pose_arr[i,5]
				# reflect left right with 50% probability
				if np.random.rand() < 0.5:
					image = np.fliplr(image)
					if self.pose_dim > 1:
						pose_arr[i,5] = -pose_arr[i,5]
				# reflect up down with 50% probability
				if np.random.rand() < 0.5:
					image = np.flipud(image)
					if self.pose_dim > 1:
						pose_arr[i,4] = -pose_arr[i,4]
				im_arr[i,:,:,0] = image

		return im_arr, pose_arr

	def __iter__(self):
		"""
		Returns a new minibatch of data with each call.
		Yields:
			tuple: The next minibatch which includes both features and labels.
		"""
		im_arr = None
		pose_arr = None
		label_arr = None
		for i in range(self.start, self.ndata, self.be.bsz):
			start_ind = 0
			while start_ind < self.be.bsz:
				# choose a random file and get the indices corresponding to it
				index = np.random.choice(len(self.im_filenames))
				indices = self.indices[self.im_filenames[index]]
				np.random.shuffle(indices)

				# get the corresponding data from that file
		                print(self.im_filenames[index])
				file_im_data = np.load(os.path.join(self.dataset_dir, self.im_filenames[index]))['arr_0'][indices]
				file_pose_data = self._read_pose_data(np.load(os.path.join(self.dataset_dir, self.pose_filenames[index]))['arr_0'][indices], self.input_data_mode)
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
				if self.distort:
					file_im_data, file_pose_data = self._distort(file_im_data, file_pose_data)
				
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
		
			# normalize images and poses
			im_arr = (im_arr - self.im_mean) / self.im_std
			pose_arr = (pose_arr - self.pose_mean[2:3]) / self.pose_std[2:3]
			# im_arr = im_arr - self.im_mean
			# pose_arr = pose_arr - self.pose_mean[:pose_arr.shape[1]]

			# now flatten the image array for neon backend
			im_arr_flat = im_arr.reshape((self.be.bsz, self.im_width * self.im_height * self.im_channels))

			# load the data into device memory and perform faster transpose using neon backend
			im_arr_dev = self.be.array(im_arr_flat, persist_values=False)
			pose_arr_dev = self.be.array(pose_arr, persist_values=False)

			if self.make_onehot:
				label_arr_dev = self.be.array(label_arr.reshape((1, -1)), dtype=np.int32, persist_values=False)
			else:
				label_arr_dev = self.be.array(label_arr, persist_values=False)

			im_arr_buf = self.be.iobuf(im_arr_flat.shape[1], dtype=np.float32, persist_values=False)
			pose_arr_buf = self.be.iobuf(pose_arr.shape[1], dtype=np.float32, persist_values=False)

			if self.make_onehot:
				label_arr_buf = self.be.iobuf(self.nclass, persist_values=False)
			else:
				label_arr_buf = self.be.iobuf(label_arr.shape[1:], persist_values=False)

			self.transpose_func(im_arr_dev, im_arr_buf)
			self.transpose_func(pose_arr_dev, pose_arr_buf)

			if self.make_onehot:
				self.onehot_func(label_arr_dev, label_arr_buf)
			else:
				self.transpose_func(label_arr_dev, label_arr_buf)

			# yield
			yield (im_arr_buf, pose_arr_buf) , label_arr_buf


