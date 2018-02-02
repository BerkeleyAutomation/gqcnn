import logging

from gqcnn.model.tf.network_tf import GQCNNTF
from gqcnn.model.neon.network_neon import GQCNNNeon
 
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
