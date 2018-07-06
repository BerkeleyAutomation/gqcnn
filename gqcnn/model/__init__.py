import logging

from gqcnn.model.tf.network_tf import GQCNNTF
from gqcnn.model.tf.fc_network_tf import FCGQCNNTF
 
def get_gqcnn_model(backend='tf'):
    # return desired GQCNN instance based on backend
    if backend == 'tf':
        logging.info('Initializing GQCNN with Tensorflow as backend...')
        return GQCNNTF
    else:
        raise ValueError('Invalid backend: {}'.format(backend))

def get_fc_gqcnn_model(backend='tf'):
    # return desired Fully-Convolutional GQCNN instance based on backend
    if backend == 'tf':
        logging.info('Initializing FC-GQCNN with Tensorflow as backend...')
        return FCGQCNNTF
    else:
        raise ValueError('Invalid backend: {}'.format(backend))
