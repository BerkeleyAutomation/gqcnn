Dex-Net 2.0
~~~~~~~~~~~
Here are the lowest error rates (%) we have achieved on the `Dex-Net-Large`_ dataset on randomized validation sets using various splitting rules:

.. image:: ../images/dexnet_benchmark.png
   :height: 800px
   :width: 800 px
   :scale: 100 %
   :align: center

The lower error rates were achieved by training model GQ from the RSS paper for additional epochs.
These models can be found in our `model zoo`_.

.. _model zoo: https://berkeley.box.com/s/szbchyt3tou9e4ct6dz8c5v99vhx0s84

We believe grasping performance on the physical robot can be improved if these validation error rates can be further reduced by modifications to the network architecture and optimization.
If you achieve superior numbers on a randomized validation set, please email Jeff Mahler (jmahler@berkeley.edu) with the subject "Dex-Net 2.0 Benchmark Submission" and we will consider testing on our ABB YuMi.

.. _Dex-Net-Large: https://berkeley.box.com/s/pub2x8mtwhrzppr11nee0q6hcx0rm32w


