"""
High-level control function to choose which instance of GQCNN Trainer 
to use based on desired backend.
Author: Vishal Satish  
"""

from gqcnn.training import GQCNNTrainerTF, GQCNNTrainerTF

def get_gqcnn_trainer(backend='tf'):
	# return desired GQCNN training instance based on backend
	if backend == 'tf':
		return GQCNNTrainerTF
	elif backend == 'neon':
		return GQCNNTrainerNeon
	else:
		raise ValueError('Invalid backend: {}'.format(backend))