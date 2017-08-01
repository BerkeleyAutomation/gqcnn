"""
Script to calculate top activations for neurons in convolutional layers of a trained GQ-CNN model.

Author: Vishal Satish 
"""

from gqcnn import GQCNN, ImageMode, ImageFileTemplates, InputDataMode, Visualizer as vis2d
from autolab_core import YamlConfig, Box
import autolab_core.utils as utils
from perception import DepthImage

import os
import gc 
import logging
import IPython
import cPickle as pickle
import numpy as np
import time

class Patch(object):
	def __init__(self, resp=0, file_num=0, im_num=0, i=None, j=None, receptive_field=0, patch=None, image=None):
		self.resp = resp
		self.file_num = file_num
		self.im_num = im_num
		self.i = i
		self.j = j
		self.receptive_field = receptive_field
		self.data = patch
		self.image = image

class TopPatches(object):
	""" Store the patches with the top responses """
	def __init__(self, num_patches, w):
		self.patches = [Patch(patch=np.zeros([w,w])) for i in range(num_patches)]
		self.size = 0

	@property
	def responses(self):
		return np.array([p.resp for p in self.patches])

	def replace_if_larger(self, patch):
		smallest_response = np.min(self.responses)
		if patch.resp > smallest_response:
			smallest_index = np.where(self.responses == smallest_response)[0][0]
			self.patches[smallest_index] = patch
			if smallest_response == 0:
				self.size += 1
			return True
		return False

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
	logging.info('Reading in analysis configurations')
	cfg = YamlConfig('cfg/tools/calc_max_activations.yaml')
	dataset_dir = cfg['dataset_dir']
	model_dir = cfg['model_dir']
	output_dir = cfg['output_dir']

	layer_data = cfg['layers']
	layers = layer_data.keys()

	image_mode = cfg['image_mode']
	data_split = cfg['data_split']
	pose_data_format = cfg['pose_data_format']

	num_top_responses = cfg['num_top_responses']
	num_responses_to_check = cfg['num_responses_to_check']
	dirty_patch_thresh = cfg['dirty_patch_thresh']

	debug = cfg['debug']
	debug_num_files = cfg['debug_num_files']

	vis_config = cfg['vis']
	vis_patches = vis_config['vis_patches']

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
   
	# save the file indices used for feature generation
	with open(os.path.join(output_dir, 'feature_generation_indices.pkl'), 'wb') as output_file:
		pickle.dump(validation_file_indices, output_file)

	# load the model into a GQ-CNN
	gqcnn = GQCNN.load(model_dir)
	gqcnn.open_session()
	logging.info('Created GQ-CNN Model')

	# get receptive field sizes
	logging.info('Calculating receptive field sizes')
	layers.sort()
	receptive_field_sizes = {}
	downsampling_rates = {}
	cur_receptive_field = 0
	cur_downsampling = 1
	for conv_layer in layers:
		cur_receptive_field += cur_downsampling * (layer_data[conv_layer]['filt_dim'])
		receptive_field_sizes[conv_layer] = cur_receptive_field
		downsampling_rates[conv_layer] = cur_downsampling
		cur_downsampling *= layer_data[conv_layer]['pool_size']

	###FOR NOW WE HARDCODE RECEPTIVE FIELDS###
	receptive_field_sizes['conv1_1'] = 7
	receptive_field_sizes['conv1_2'] = 11
	receptive_field_sizes['conv2_1'] = 17
	receptive_field_sizes['conv2_2'] = 22
	IPython.embed()

	# iterate through each layer and calculate the max activations for it
	for layer in layers:
		# dictionary of TopPatches objects representing for each filter
		top_patches = {}

		# load each image file and its corresponding pose file if there are validation indices in it
		for x in range(len(im_filenames)):
			filename = im_filenames[x]
			if filename in validation_file_indices.keys():
				image_arr = np.load(os.path.join(dataset_dir, filename))['arr_0'][validation_file_indices[filename]]
				pose_arr = read_pose_data(np.load(os.path.join(dataset_dir, pose_filenames[x]))['arr_0'][validation_file_indices[filename]], pose_data_format)
				
				# pull features from the GQ-CNN
				logging.info('Computing max activations for layer {} with image file {}'.format(layer, filename))
				
				featurization_start_time = time.time()
				feature_maps = gqcnn.featurize(image_arr, pose_arr, layer)[:len(image_arr)]
				logging.info('Featurization took {} seconds'.format(time.time() - featurization_start_time))

				receptive_field = receptive_field_sizes[layer]
				layer_downsampling = downsampling_rates[layer]

				num_filter = feature_maps.shape[3]

				for i in range(num_filter):
					neuron_name = '{}_{}'.format(layer, i)
					if neuron_name not in top_patches.keys():
						top_patches[neuron_name] = TopPatches(num_top_responses, receptive_field)

					# get the top responses
					neuron_responses = feature_maps[:,:,:,i]
					sorted_responses = np.sort(neuron_responses, axis=None)
					top_responses = sorted_responses[-num_responses_to_check:]

					# cycle through the top responses and add patches if they are good
					for response in top_responses[::-1]:
						# if the largest response in what's left is less than the smallest response in our set of saved resposnes, just break out and stop checking
						if response <= np.min(top_patches[neuron_name].responses):
							# logging.info("Breaking, Size: {}, Response: {}".format(top_patches[neuron_name].size, response))
							break

						# get the indices and location of the patch
						indices = np.where(feature_maps == response)
						if len(indices[0]) == 1:
							num_top_response_positions = 1
						else:
							num_top_response_positions = 0
							for x in range(len(indices[0]) - 1):
								current = indices[0][x]
								nxt = indices[0][x + 1]
								num_top_response_positions += 1
								if current != nxt:
									break
						im_index = indices[0][0]
						source_px_i = layer_downsampling * indices[1][:num_top_response_positions]
						source_px_j = layer_downsampling * indices[2][:num_top_response_positions]
						
						min_row = max(0, source_px_i[0]-(receptive_field / 2))
						min_col = max(0, source_px_j[0]-(receptive_field / 2))
						max_row = min(image_arr.shape[1], source_px_i[0]+ (receptive_field / 2) +1)
						max_col = min(image_arr.shape[2], source_px_j[0]+ (receptive_field / 2) +1)

						patch = image_arr[im_index, min_row:max_row, min_col:max_col, 0]

						# get rid of bad patches
						if np.min(patch) < dirty_patch_thresh:
							num_bad_patches += 1
							logging.info('Found {} bad patches'.format(num_bad_patches))
						else:
							replaced = top_patches[neuron_name].replace_if_larger(Patch(response, filename[-9:-4], im_index, source_px_i, source_px_j, receptive_field, patch.copy(), image_arr[im_index]))

		# save the output
		logging.info('Saving images for layer {}'.format(layer))
		layer_dir = os.path.join(output_dir, layer)
		vis2d.figure()
		if not os.path.exists(layer_dir):
			os.mkdir(layer_dir)
		for neuron in top_patches:
			# create a directory for the neuron
			neuron_dir = os.path.join(layer_dir, neuron)
			if not os.path.exists(neuron_dir):
				os.mkdir(neuron_dir)
			patches = top_patches[neuron]
			i = 0
			for patch in patches.patches:
				logging.info('Saving image for layer {} filter {} neuron {}'.format(layer, neuron, i))
				vis2d.clf()
				boxes = []
				if patch.i is None or patch.j is None:
					logging.info('Found bad batch index data')
					i +=1
					continue
				for I, J in zip(patch.i, patch.j):
					min_x = I - (patch.receptive_field / 2)
					min_y = J - (patch.receptive_field / 2)
					max_x = I + (patch.receptive_field / 2)
					max_y = J + (patch.receptive_field / 2)
					boxes.append(Box(np.asarray([min_x, min_y]), np.asarray([max_x, max_y]), 'None'))

				vis2d.imshow(DepthImage(patch.image))
				if len(boxes) > 1:
					logging.info('Multiple Boxes')
				for box in boxes:
					vis2d.box(box)

				if vis_patches:
					vis2d.show()

				# save figure
				vis2d.savefig(os.path.join(neuron_dir, "patch_{}.png".format(i)))
			
				i += 1

	# close the gqcnn session
	gqcnn.close_session()
