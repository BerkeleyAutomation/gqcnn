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
import os
import sys
import logging
import subprocess
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

TF_MIN_VERSION = '1.10.0'
TF_MAX_VERSION = '1.13.1'

# set up logger
logging.basicConfig() # configure the root logger
logger = logging.getLogger('setup.py')
logger.setLevel(logging.INFO)

def get_tf_dep():
    # check whether or not the Nvidia driver and GPUs are available and add the corresponding Tensorflow dependency
    tf_dep = 'tensorflow>={},<={}'.format(TF_MIN_VERSION, TF_MAX_VERSION)
    try:
        gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv']).decode().strip().split('\n')[1:]
        if len(gpus) > 0:
            tf_dep = 'tensorflow-gpu>={},<={}'.format(TF_MIN_VERSION, TF_MAX_VERSION)
        else:
            logger.warning('Found Nvidia device driver but no devices...installing Tensorflow for CPU.')
    except OSError:
        logger.warning('Could not find Nvidia device driver...installing Tensorflow for CPU.')
    return tf_dep

#TODO(vsatish): Use inheritance here
class DevelopCmd(develop):
    user_options_custom = [
        ('docker', None, 'installing in Docker'),
    ]
    user_options = getattr(develop, 'user_options', []) + user_options_custom

    def initialize_options(self):
        develop.initialize_options(self)

        # initialize options
        self.docker = False

    def finalize_options(self):
        develop.finalize_options(self)

    def run(self):
        # install Tensorflow dependency
        if not self.docker:
            tf_dep = get_tf_dep()
            subprocess.Popen([sys.executable, '-m', 'pip', 'install', tf_dep]).wait()
        else:
            # if we're using docker, this will already have been installed explicitly through the correct {cpu/gpu}_requirements.txt; there is no way to check for CUDA/GPUs at docker build time because there is no easy way to set the nvidia runtime
            logger.warning('Omitting Tensorflow dependency because of Docker installation.') #TODO(vsatish): Figure out why this isn't printed

        # run installation
        develop.run(self)

class InstallCmd(install, object):
    user_options_custom = [
        ('docker', None, 'installing in Docker'),
    ]
    user_options = getattr(install, 'user_options', []) + user_options_custom

    def initialize_options(self):
        install.initialize_options(self)

        # initialize options
        self.docker = False

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        # install Tensorflow dependency
        if not self.docker:
            tf_dep = get_tf_dep()
            subprocess.Popen([sys.executable, '-m', 'pip', 'install', tf_dep]).wait()
        else:
            # if we're using docker, this will already have been installed explicitly through the correct {cpu/gpu}_requirements.txt; there is no way to check for CUDA/GPUs at docker build time because there is no easy way to set the nvidia runtime
            logger.warning('Omitting Tensorflow dependency because of Docker installation.') #TODO(vsatish): Figure out why this isn't printed

        # run installation
        install.run(self)

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

exec(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gqcnn/version.py')).read())

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
      cmdclass={
        'install': InstallCmd,
        'develop': DevelopCmd
      }
)

