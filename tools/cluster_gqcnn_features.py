"""
Script for clustering together gqcnn features using k-means and finding image closest to centroid of each cluster.
Author: Vishal Satish
"""

import os
import logging
import cPickle as pkl
import numpy as np
import IPython
import time
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from autolab_core import YamlConfig
from perception import DepthImage
from gqcnn import Visualizer as vis2d

if __name__ == "__main__":
	# setup logger
	logging.getLogger().setLevel(logging.INFO)
	
	# read config parameters
	logging.info('Reading in cluster configurations')
	cfg = YamlConfig('cfg/tools/cluster_gqcnn_features.yaml')
	dataset_dir = cfg['dataset_dir']
	feature_dir = cfg['feature_dir']
	output_dir = cfg['output_dir']

	layers = cfg['layers']

	k_means_cfg = cfg['kmeans']
	k_values = k_means_cfg['k_values']
	num_samples_per_centroid = k_means_cfg['num_samples_per_centroid']

	im_height = cfg['im_height']
	im_width = cfg['im_width']
	im_channels = cfg['im_channels']

	vis = cfg['vis']

	# load feature generation file indices
	logging.info('Loading Feature Generation Indices')
	feature_generation_indices = pkl.load(open(os.path.join(feature_dir, 'feature_generation_indices.pkl'), 'rb'))
	feature_keys = feature_generation_indices.keys()    

	# sort the keys so there will be a one-to-one mapping with the features
	feature_keys.sort(key = lambda x: int(x[-9:-4]))

	# iterate through the layers
	for layer in layers:
		logging.info('Finding representative images for layer {}'.format(layer))
		
		# load the features
		logging.info('Loading features')
		feature_path = os.path.join(feature_dir, "tf_features_{}.npz".format(layer))
		features = np.load(feature_path)['arr_0']

		# run k-means
		logging.info('Computing K-Means')		
		k_means = {}
		for k in k_values:
			logging.info('Running K-Means for k = {}'.format(k))
			start_time = time.time()
			kmeans = KMeans(n_clusters=k, n_jobs=-1).fit(features)
			logging.info('K-Means Algorithm took {} seconds'.format(time.time() - start_time))
			k_means[k] = kmeans

		# find the feature_maps closest to the centroids of the clusters
		logging.info('Finding images closest to centroids')
		for k in k_means:
			# dict to hold top n closest feature indices to each centroid
			feature_indices = {}
			for x in range(len(k_means[k].cluster_centers_)):
				feature_indices["c_{}".format(x)] = []

			# get top n closes feature indices
			logging.info('Finding images for k = {}'.format(k))
			start_time = time.time()

			# make copy of original features arr
			feature_copy = features[:]

			for x in range(num_samples_per_centroid):
				# get closest feature indices
				closest_feature_map_indices, _ = pairwise_distances_argmin_min(k_means[k].cluster_centers_, features)
				# add them to dict
				for i in range(len(closest_feature_map_indices)):
					feature_indices["c_{}".format(i)].append(closest_feature_map_indices[i])
				# remove those features
				features = np.delete(features, closest_feature_map_indices, 0)

			# reset feature arr back to normal
			features = feature_copy[:]

			logging.info('Finding closest features to centroids took {} seconds'.format(time.time() - start_time))
			# convert feature indices to image file indices
			converted_feature_indices = {}
			logging.info('Converting feature indices to image file indices')
			for cluster in feature_indices:
				converted_indices = []
				for feature_map_index in feature_indices[cluster]:
					counter = 0
					for key in feature_keys:
						num_indices = len(feature_generation_indices[key])
						if (feature_map_index - counter) <= num_indices:
							converted_indices.append((key, feature_generation_indices[key][(feature_map_index - counter) - 1]))
							break
						else:
							counter += len(feature_generation_indices[key])
				converted_feature_indices[cluster] = converted_indices
			### get images ###

			logging.info('Getting images')
			# allocate image tensor
			images = np.zeros((len(converted_feature_indices), num_samples_per_centroid, im_width, im_height, im_channels))

			# iterate through converted indices
			cluster_i = 0
			for cluster in converted_feature_indices:
				image_i = 0
				for filename, index in converted_feature_indices[cluster]:
					im_arr = np.load(os.path.join(dataset_dir, filename))['arr_0'][index]

					# add image data to tensor
					images[cluster_i][image_i] = im_arr
					image_i += 1
				cluster_i += 1

			# save images
			k_dir = os.path.join(output_dir, layer, "k={}".format(k))

			# save the images for each cluster
			for cluster in range(len(images)):
				cluster_dir = os.path.join(k_dir, "cluster_{}".format(cluster))
				if not os.path.exists(cluster_dir):
					os.makedirs(cluster_dir)

				vis2d.figure()
				i = 0
				for image in images[cluster]:
					vis2d.clf()
					vis2d.imshow(DepthImage(image))

					if vis:
						vis2d.show()

					# save image
					logging.info('Saving image {} in cluster {} for k = {} for layer {}'.format(i, cluster, k, layer))
					vis2d.savefig(os.path.join(cluster_dir, "image {}.png").format(i))
					i += 1