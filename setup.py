# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Setup of gqcnn python codebase.

Author 
------
Jeff Mahler & Vishal Satish
"""
import logging
from setuptools import setup, find_packages
from subprocess import check_output

# set up logger
logging.basicConfig() # configure the root logger
logger = logging.getLogger('setup.py')
logger.setLevel(logging.INFO)

requirements = [
    'autolab-core',
    'autolab-perception',
    'visualization',
    'numpy>=1.14.0',
    'scipy',
    'matplotlib<3.0.0',
    'opencv-python',
    'scikit-image<0.15.0',
    'scikit-learn',
    'psutil',
    'gputil'
]

# check whether or not the Nvidia driver and GPUs are available and add the corresponding Tensorflow dependency
tf_min_version = '1.10.0'
tf_max_version = '1.13.1'
tf_dep = 'tensorflow>={},<={}'.format(tf_min_version, tf_max_version)
try:
    gpus = check_output(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv']).decode().strip().split('\n')[1:]
    if len(gpus) > 0:
        tf_dep = 'tensorflow-gpu>={},<={}'.format(tf_min_version, tf_max_version)
    else:
        logger.warning('Found Nvidia device driver but no devices...installing Tensorflow for CPU.')
except OSError:
    logger.warning('Could not find Nvidia device driver...installing Tensorflow for CPU.')
requirements.append(tf_dep)

exec(open('gqcnn/version.py').read())

setup(name='gqcnn', 
      version=__version__, 
      description='Project code for running Grasp Quality Convolutional Neural Networks', 
      author='Vishal Satish', 
      author_email='vsatish@berkeley.edu', 
      license = 'Berkeley Copyright',
      url = 'https://github.com/BerkeleyAutomation/gqcnn',
      keywords = 'robotics grasping vision deep learning',
      classifiers = [
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 2.7 :: Only',
          'Natural Language :: English',
          'Topic :: Scientific/Engineering'
      ],      
      packages=find_packages(), 
      install_requires = requirements,
      extras_require = { 'docs' : [
          'sphinx',
          'sphinxcontrib-napoleon',
          'sphinx_rtd_theme'
      ],
      },
)

