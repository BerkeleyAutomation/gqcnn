from gqcnn.training.tf.trainer_tf import GQCNNTrainerTF

def get_gqcnn_trainer(backend='tf'):
    # return desired GQCNN training instance based on backend
    if backend == 'tf':
        return GQCNNTrainerTF
    else:
        raise ValueError('Invalid backend: {}'.format(backend))
