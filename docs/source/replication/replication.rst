Replicating Results
~~~~~~~~~~~~~~~~~~~
Numerous publications in the larger `Dex-Net`_ project utilize GQ-CNNs, particularly `Dex-Net 2.0`_, `Dex-Net 2.1`_, `Dex-Net 3.0`_, `Dex-Net 4.0`_, and `FC-GQ-CNN`_. One of the goals of the `gqcnn` library is to provide the necessary code and instructions for replicating the results of these papers. Thus, we have provided a number of replication scripts under the `scripts/` directory.

There are two ways to replicate results:

#. **Use a pre-trained model:** Download a pre-trained GQ-CNN model and run an example policy.
#. **Train from scratch:** Download the raw dataset, train a GQ-CNN model, and run an example policy using the model you just trained.

**We highly encourage method 1.** Note that method 2 is computationally expensive as training takes roughly 24 hours on a Nvidia Titan Xp GPU. Furthermore, the raw datasets are fairly large in size. 

Please keep in mind that GQ-CNN models are sensitive to the following parameters used during dataset generation:
   #. The robot gripper
   #. The depth camera
   #. The distance between the camera and workspace.

As a result, we cannot guarantee performance of our pre-trained models on other physical setups.

For more information about the pre-trained models and sample inputs for the example policy, see :ref:`pre-trained-models` and :ref:`sample-inputs`.

Using a Pre-trained Model
=========================
First download the pre-trained models. ::

    $ ./scripts/downloads/models/download_models.sh

Dex-Net 2.0
"""""""""""
Evaluate the pre-trained GQ-CNN model. ::

    $ ./scripts/policies/run_all_dex-net_2.0_examples.sh

Dex-Net 2.1
"""""""""""
Evaluate the pre-trained GQ-CNN model. ::

    $ ./scripts/policies/run_all_dex-net_2.1_examples.sh

Dex-Net 3.0
"""""""""""
Evaluate the pre-trained GQ-CNN model. ::

    $ ./scripts/policies/run_all_dex-net_3.0_examples.sh

Dex-Net 4.0
"""""""""""
To evaluate the pre-trained **parallel jaw** GQ-CNN model. ::

    $ ./scripts/policies/run_all_dex-net_4.0_pj_examples.sh

To evaluate the pre-trained **suction** GQ-CNN model. ::

    $ ./scripts/policies/run_all_dex-net_4.0_suction_examples.sh

FC-GQ-CNN
"""""""""""
To evaluate the pre-trained **parallel jaw** FC-GQ-CNN model. ::

    $ ./scripts/policies/run_all_dex-net_4.0_fc_pj_examples.sh

To evaluate the pre-trained **suction** FC-GQ-CNN model. ::

    $ ./scripts/policies/run_all_dex-net_4.0_fc_suction_examples.sh



Training from Scratch
=====================

Dex-Net 2.0
"""""""""""
First download the appropriate dataset. ::

    $ ./scripts/downloads/datasets/download_dex-net_2.0.sh

Then train a GQ-CNN from scratch. ::

    $ ./scripts/training/train_dex-net_2.0.sh

Finally, evaluate the trained GQ-CNN. :: 

    $ ./scripts/policies/run_all_dex-net_2.0_examples.sh 

Dex-Net 2.1
"""""""""""
First download the appropriate dataset. ::

    $ ./scripts/downloads/datasets/download_dex-net_2.1.sh

Then train a GQ-CNN from scratch. ::

    $ ./scripts/training/train_dex-net_2.1.sh

Finally, evaluate the trained GQ-CNN. :: 

    $ ./scripts/policies/run_all_dex-net_2.1_examples.sh 

Dex-Net 3.0
"""""""""""
First download the appropriate dataset. ::

    $ ./scripts/downloads/datasets/download_dex-net_3.0.sh

Then train a GQ-CNN from scratch. ::

    $ ./scripts/training/train_dex-net_3.0.sh

Finally, evaluate the trained GQ-CNN. :: 

    $ ./scripts/policies/run_all_dex-net_3.0_examples.sh 

Dex-Net 4.0
"""""""""""
To replicate the `Dex-Net 4.0`_ **parallel jaw** results, first download the appropriate dataset. ::

    $ ./scripts/downloads/datasets/download_dex-net_4.0_pj.sh

Then train a GQ-CNN from scratch. ::

    $ ./scripts/training/train_dex-net_4.0_pj.sh

Finally, evaluate the trained GQ-CNN. :: 

    $ ./scripts/policies/run_all_dex-net_4.0_pj_examples.sh

To replicate the `Dex-Net 4.0`_ **suction** results, first download the appropriate dataset. ::

    $ ./scripts/downloads/datasets/download_dex-net_4.0_suction.sh

Then train a GQ-CNN from scratch. ::

    $ ./scripts/training/train_dex-net_4.0_suction.sh

Finally, evaluate the trained GQ-CNN. :: 

    $ ./scripts/policies/run_all_dex-net_4.0_suction_examples.sh

FC-GQ-CNN
"""""""""""
To replicate the `FC-GQ-CNN`_ **parallel jaw** results, first download the appropriate dataset. ::

    $ ./scripts/downloads/datasets/download_dex-net_4.0_fc_pj.sh

Then train a FC-GQ-CNN from scratch. ::

    $ ./scripts/training/train_dex-net_4.0_fc_pj.sh

Finally, evaluate the trained FC-GQ-CNN. :: 

    $ ./scripts/policies/run_all_dex-net_4.0_fc_pj_examples.sh

To replicate the `FC-GQ-CNN`_ **suction** results, first download the appropriate dataset. ::

    $ ./scripts/downloads/datasets/download_dex-net_4.0_fc_suction.sh

Then train a FC-GQ-CNN from scratch. ::

    $ ./scripts/training/train_dex-net_4.0_fc_suction.sh

Finally, evaluate the trained FC-GQ-CNN. :: 

    $ ./scripts/policies/run_all_dex-net_4.0_fc_suction_examples.sh

.. _Dex-Net: https://berkeleyautomation.github.io/dex-net/
.. _Dex-Net 2.0: https://berkeleyautomation.github.io/dex-net/#dexnet_2
.. _Dex-Net 2.1: https://berkeleyautomation.github.io/dex-net/#dexnet_21
.. _Dex-Net 3.0: https://berkeleyautomation.github.io/dex-net/#dexnet_3
.. _Dex-Net 4.0: https://berkeleyautomation.github.io/dex-net/#dexnet_4
.. _FC-GQ-CNN: https://berkeleyautomation.github.io/fcgqcnn 
.. _gqcnn: https://github.com/BerkeleyAutomation/gqcnn

