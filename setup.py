# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents).
All Rights Reserved. Permission to use, copy, modify, and distribute this
software and its documentation for educational, research, and not-for-profit
purposes, without fee and without a signed licensing agreement, is hereby
granted, provided that the above copyright notice, this paragraph and the
following two paragraphs appear in all copies, modifications, and
distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Setup of `gqcnn` Python codebase.

Author
------
Vishal Satish & Jeff Mahler
"""
import logging
import os
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess
import sys

TF_MAX_VERSION = "1.15.0"

# Set up logger.
logging.basicConfig()  # Configure the root logger.
logger = logging.getLogger("setup.py")
logger.setLevel(logging.INFO)


def get_tf_dep():
    # Check whether or not the Nvidia driver and GPUs are available and add the
    # corresponding Tensorflow dependency.
    tf_dep = "tensorflow<={}".format(TF_MAX_VERSION)
    try:
        gpus = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=gpu_name",
             "--format=csv"]).decode().strip().split("\n")[1:]
        if len(gpus) > 0:
            tf_dep = "tensorflow-gpu<={}".format(TF_MAX_VERSION)
        else:
            no_device_msg = ("Found Nvidia device driver but no"
                             " devices...installing Tensorflow for CPU.")
            logger.warning(no_device_msg)
    except OSError:
        no_driver_msg = ("Could not find Nvidia device driver...installing"
                         " Tensorflow for CPU.")
        logger.warning(no_driver_msg)
    return tf_dep


# TODO(vsatish): Use inheritance here.
class DevelopCmd(develop):
    user_options_custom = [
        ("docker", None, "installing in Docker"),
    ]
    user_options = getattr(develop, "user_options", []) + user_options_custom

    def initialize_options(self):
        develop.initialize_options(self)

        # Initialize options.
        self.docker = False

    def finalize_options(self):
        develop.finalize_options(self)

    def run(self):
        # Install Tensorflow dependency.
        if not self.docker:
            tf_dep = get_tf_dep()
            subprocess.Popen([sys.executable, "-m", "pip", "install",
                              tf_dep]).wait()
        else:
            # If we're using Docker, this will already have been installed
            # explicitly through the correct `{cpu/gpu}_requirements.txt`;
            # there is no way to check for CUDA/GPUs at Docker build time
            # because there is no easy way to set the Nvidia runtime.
            # TODO(vsatish): Figure out why this isn't printed.
            skip_tf_msg = ("Omitting Tensorflow dependency because of Docker"
                           " installation.")
            logger.warning(skip_tf_msg)

        # Run installation.
        develop.run(self)


class InstallCmd(install, object):
    user_options_custom = [
        ("docker", None, "installing in Docker"),
    ]
    user_options = getattr(install, "user_options", []) + user_options_custom

    def initialize_options(self):
        install.initialize_options(self)

        # Initialize options.
        self.docker = False

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        # Install Tensorflow dependency.
        if not self.docker:
            tf_dep = get_tf_dep()
            subprocess.Popen([sys.executable, "-m", "pip", "install",
                              tf_dep]).wait()
        else:
            # If we're using Docker, this will already have been installed
            # explicitly through the correct `{cpu/gpu}_requirements.txt`;
            # there is no way to check for CUDA/GPUs at Docker build time
            # because there is no easy way to set the Nvidia runtime.
            # TODO (vsatish): Figure out why this isn't printed.
            skip_tf_msg = ("Omitting Tensorflow dependency because of Docker"
                           " installation.")
            logger.warning(skip_tf_msg)

        # Run installation.
        install.run(self)


requirements = [
    "autolab-core", "autolab-perception", "visualization", "numpy", "scipy",
    "matplotlib", "opencv-python", "scikit-learn", "scikit-image", "psutil",
    "gputil"
]

exec(
    open(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     "gqcnn/version.py")).read())

setup(
    name="gqcnn",
    version=__version__,  # noqa F821
    description=("Project code for running Grasp Quality Convolutional"
                 " Neural Networks"),
    author="Vishal Satish",
    author_email="vsatish@berkeley.edu",
    license="Berkeley Copyright",
    url="https://github.com/BerkeleyAutomation/gqcnn",
    keywords="robotics grasping vision deep learning",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Natural Language :: English",  # yapf: disable
        "Topic :: Scientific/Engineering"
    ],
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "docs": ["sphinx", "sphinxcontrib-napoleon", "sphinx_rtd_theme"],
    },
    cmdclass={
        "install": InstallCmd,
        "develop": DevelopCmd
    })
