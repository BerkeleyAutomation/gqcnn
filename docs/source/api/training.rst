Training
========

GQ-CNN training classes are **never accessed directly**, but through a lightweight factory function that returns the corresponding class depending on the specified backend. ::

    $ from gqcnn import get_gqcnn_trainer
    $
    $ backend = 'tf'
    $ my_trainer = get_gqcnn_trainer(backend)(<class initializer args>)

.. autofunction:: gqcnn.get_gqcnn_trainer

GQCNNTrainerTF
~~~~~~~~~~~~~~

.. autoclass:: gqcnn.training.tf.GQCNNTrainerTF

