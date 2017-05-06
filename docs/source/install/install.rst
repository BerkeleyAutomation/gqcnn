Python Installation
~~~~~~~~~~~~~~~~~~~

Note that the `GQCNN` module is known to work for Python 2.7 and has not been tested for Python 3.

1. Clone the repository
"""""""""""""""""""""""
Clone or download our source code from `Github`_. ::

    $ git clone https://github.com/BerkeleyAutomation/gqcnn.git

.. _Github: https://github.com/BerkeleyAutomation/gqcnn

2. Run installation script
""""""""""""""""""""""""""
Change directories into the `gqcnn` repository and run ::

    $ python setup.py install

or ::

    $ pip install -r requirements.txt

Alternatively, you can run ::

    $ pip install /path/to/gqcnn

to install `gqcnn` from anywhere.
This will install `gqcnn` in your current Python environment.

Dependencies
~~~~~~~~~~~~
The `GQCNN` module depends on the Berkeley AutoLab's `core`_ module, which can be installed using the instructions `here`_.

The `gqcnn` module's other dependencies are on `numpy`_, `scipy`_, `matplotlib`_, `tensorflow`_, `cv2`_, `skimage`_, and `sklearn`_ and should be installed automatically.
You can install these manually if you wish with
pip. ::

    $ pip install numpy
    $ pip install scipy
    $ pip install matplotlib
    $ pip install tensorflow
    $ pip install opencv-python
    $ pip install scikit-image
    $ pip install scikit-learn

However, installing our repo using `pip` will install these automatically.

.. _numpy: http://www.numpy.org/
.. _scipy: https://www.scipy/org/
.. _matplotlib: http://www.matplotlib.org/
.. _core: https://github.com/BerkeleyAutomation/core
.. _here: https://BerkeleyAutomation.github.io/core
.. _tensorflow: https://www.tensorflow.org/
.. _cv2: http://opencv.org/
.. _skimage: http://scikit-learn.org/stable/
.. _sklearn: http://scikit-image.org/

Documentation
~~~~~~~~~~~~~

Building
""""""""
Building `GQCNN`'s documentation requires a few extra dependencies --
specifically, `sphinx`_ and a few plugins.

.. _sphinx: http://www.sphinx-doc.org/en/1.4.8/

To install the dependencies required, simply run ::

    $ pip install -r docs_requirements.txt

Then, go to the `docs` directory and run ``make`` with the appropriate target.
For example, ::

    $ cd docs/
    $ make html

will generate a set of web pages. Any documentation files
generated in this manner can be found in `docs/build`.

Deploying
"""""""""
To deploy documentation to the Github Pages site for the repository,
simply push any changes to the documentation source to master
and then run ::

    $ . gh_deploy.sh

from the `docs` folder. This script will automatically checkout the
``gh-pages`` branch, build the documentation from source, and push it
to Github.

