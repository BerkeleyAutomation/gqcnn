GQ-CNN
======

GQ-CNN and FC-GQ-CNN classes are **never accessed directly**, but through a lightweight factory function that returns the corresponding class depending on the specified backend. ::

    $ from gqcnn import get_gqcnn_model
    $
    $ backend = 'tf'
    $ my_gqcnn = get_gqcnn_model(backend)(<class initializer args>)

.. autofunction:: gqcnn.get_gqcnn_model

.. autofunction:: gqcnn.get_fc_gqcnn_model

GQCNNTF
~~~~~~~

Tensorflow implementation of GQ-CNN model.

.. autoclass:: gqcnn.model.tf.GQCNNTF
    :exclude-members: init_mean_and_std,
                      set_base_network,
                      init_weights_file,
                      initialize_network,
                      set_batch_size,
                      set_im_mean,
                      get_im_mean,
                      set_im_std,
                      get_im_std,
                      set_pose_mean,
                      get_pose_mean,
                      set_pose_std,
                      get_pose_std,
                      set_im_depth_sub_mean,
                      set_im_depth_sub_std,
                      add_softmax_to_output,
                      add_sigmoid_to_output,
                      update_batch_size,
                      
FCGQCNNTF
~~~~~~~~~

Tensorflow implementation of FC-GQ-CNN model.

.. autoclass:: gqcnn.model.tf.FCGQCNNTF
    :exclude-members: __init__
