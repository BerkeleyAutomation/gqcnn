"""
Script for visualizing the filters of a GQ-CNN.

Author: Vishal Satish
"""
from gqcnn import GQCNN
from autolab_core import YamlConfig

import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import IPython

if __name__ == '__main__':
	# load configurations
	cfg = YamlConfig('cfg/tools/gqcnn_filter_visualizer.yaml')
	model_dir = cfg['model_dir']
	output_dir = cfg['output_dir']
	models = cfg['models'].keys()

	# setup logger
	logging.getLogger().setLevel(logging.INFO)	

	# go through every model 
	for model in models:
		model_cfg = cfg['models'][model]

		# load model into a GQCNN
		gqcnn = GQCNN.load(os.path.join(model_dir, model))

		logging.info('GQCNN model {} loaded!'.format(model))

		save_dir = os.path.join(output_dir, model)
		os.mkdir(save_dir)

		# visualize the conv1_1 filters
		if model_cfg['conv1_1']:
			conv1_1W = gqcnn.get_weights().conv1_1W
			conv1_1b = gqcnn.get_weights().conv1_1b

			num_filt = conv1_1W.shape[3]
			IPython.embed()
			d = int(np.ceil(np.sqrt(num_filt)))

			plt.clf()
			for i in range(num_filt):
				plt.subplot(d,d,i+1)
				plt.imshow(conv1_1W[:,:,0,i], cmap=plt.cm.gray, interpolation='nearest')
				plt.axis('off')
				plt.title('b=%.3f' %(conv1_1b[i]), fontsize=10)
			if model_cfg['show_filters']:
				plt.show()

			# save the filters
			plt.savefig(os.path.join(save_dir, 'conv1_1W_%05d.jpg'))


		# visualize the conv1_2 filters
		if model_cfg['conv1_2']:
			conv1_2W = gqcnn.get_weights().conv1_2W
			conv1_2b = gqcnn.get_weights().conv1_2b

			num_filt = conv1_2W.shape[3]
			d = int(np.ceil(np.sqrt(num_filt)))

			plt.clf()
			for i in range(num_filt):
				plt.subplot(d,d,i+1)
				plt.imshow(conv1_2W[:,:,0,i], cmap=plt.cm.gray, interpolation='nearest')
				plt.axis('off')
				plt.title('b=%.3f' %(conv1_2b[i]), fontsize=10)
			if model_cfg['show_filters']:
				plt.show()

			# save the filters
			plt.savefig(os.path.join(save_dir, 'conv1_2W_%05d.jpg'))

		# visualize the conv2_1 filters
		if model_cfg['conv2_1']:
			conv2_1W = gqcnn.get_weights().conv2_1W
			conv2_1b = gqcnn.get_weights().conv2_1b

			num_filt = conv2_1W.shape[3]
			d = int(np.ceil(np.sqrt(num_filt)))

			plt.clf()
			for i in range(num_filt):
				plt.subplot(d,d,i+1)
				plt.imshow(conv2_1W[:,:,0,i], cmap=plt.cm.gray, interpolation='nearest')
				plt.axis('off')
				plt.title('b=%.3f' %(conv2_1b[i]), fontsize=10)
			if model_cfg['show_filters']:
				plt.show()

			# save the filters
			plt.savefig(os.path.join(save_dir, 'conv2_1W_%05d.jpg'))


		# visualize the conv2_2 filters
		if model_cfg['conv2_2']:
			conv2_2W = gqcnn.get_weights().conv2_2W
			conv2_2b = gqcnn.get_weights().conv2_2b

			num_filt = conv2_2W.shape[3]
			d = int(np.ceil(np.sqrt(num_filt)))

			plt.clf()
			for i in range(num_filt):
				plt.subplot(d,d,i+1)
				plt.imshow(conv2_2W[:,:,0,i], cmap=plt.cm.gray, interpolation='nearest')
				plt.axis('off')
				plt.title('b=%.3f' %(conv2_2b[i]), fontsize=10)
			if model_cfg['show_filters']:
				plt.show()

			# save the filters
			plt.savefig(os.path.join(save_dir, 'conv2_2W_%05d.jpg'))


