""" 
High-level control function to choose which instance of GQCNN model
to use based on desired backend.
Author: Vishal Satish
"""

from gqcnn.models import GQCNNTF, GQCNNNeon

def get_gqcnn_model(backend='tf'):
	# return desired GQCNN instance based on backend
	if backend == 'tf':
		logging.info('Initializing GQCNN with Tensorflow as backend.')
		return GQCNNTF
	elif backend == 'neon':
		logging.info('Initializing GQCNN with Neon as backend.')
		return GQCNNNeon
	else:
		raise ValueError('Invalid backend: {}'.format(backend))