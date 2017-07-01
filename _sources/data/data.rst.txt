Download Link
~~~~~~~~~~~~~
GQ-CNN training datasets and pretrained network weights are available from our `data repository`_.
New datasets and models will be uploaded to this location as they become available.

.. _data repository: https://berkeley.box.com/s/p85ov4dx7vbq6y1l02gzrnsexg6yyayb

Datasets
~~~~~~~~
The available `datasets`_ include:

1) **Dex-Net-Large:** The full 6.7 million synthetic training datapoints across 1,500 object models from Dex-Net 2.0.
2) **Adv-Synth:** Synthetic training datapoints for the eight adversarial training objects (~189k datapoints).
3) **Adv-Phys:** The outcomes of executing 400 random antipodal grasps on the eight adversarial training objects with an ABB YuMi.

More details can be found in `the Dex-Net 2.0 paper`_.
The Dex-Net-Large dataset was generated from 3D models from the `KIT`_ and `3DNet`_ datasets.

.. _datasets: http://bit.ly/2rIM7Jk
.. _KIT: https://h2t-projects.webarchiv.kit.edu/Projects/ObjectModelsWebUI/
.. _3DNet: https://repo.acin.tuwien.ac.at/tmp/permanent/3d-net.org/

Models
~~~~~~
The available pre-trained `models`_ include:

1) **GQ:** The GQ-CNN trained on the full Dex-Net-Large dataset as described in `the Dex-Net 2.0 paper`_.
2) **GQ-Image-Wise:** The GQ model trained on an image-wise training and validation script for 100,000 iterations.
3) **GQ-Stable-Pose-Wise:** The GQ model trained on a stable-pose-wise training and validation script.
4) **GQ-Object-Wise:** The GQ model trained on an object-wise training and validation script.
5) **GQ-Adv:** The GQ-CNN trained on the Adv-Synth dataset as described in `the Dex-Net 2.0 paper`_.

.. _models: http://bit.ly/2tAFMko
.. _the Dex-Net 2.0 paper: https://github.com/BerkeleyAutomation/dex-net/raw/gh-pages/docs/dexnet_rss2017_final.pdf

License
~~~~~~~
The GQ-CNN model and datasets are released for unrestricted use.
The datasets are generated from 3D object models from `3DNet`_ and `the KIT Object Database`_ that may be subject to copyright.

.. _3DNet: https://repo.acin.tuwien.ac.at/tmp/permanent/3d-net.org/
.. _the KIT Object Database: https://h2t-projects.webarchiv.kit.edu/Projects/ObjectModelsWebUI/

Our understanding as researchers is that there is no restriction placed on the open release of the datasets or learned model weights, since none of the original 3D models are distributed.
If the interpretation arises that the datasets or weights are derivative works of the original copyright holder and they assert such a copyright, UC Berkeley makes no representations as to what use is allowed other than to consider our present release in the spirit of fair use in the academic mission of the university to disseminate knowledge and tools as broadly as possible without restriction. 
