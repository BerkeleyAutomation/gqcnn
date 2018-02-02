
from gqcnn.training.tf.trainer_tf import GQCNNTrainerTF
from gqcnn.training.neon.trainer_neon import GQCNNTrainerNeon

def get_gqcnn_trainer(backend='tf'):
    # return desired GQCNN training instance based on backend
    if backend == 'tf':
        return GQCNNTrainerTF
    elif backend == 'neon':
        return GQCNNTrainerNeon
    else:
        raise ValueError('Invalid backend: {}'.format(backend))
