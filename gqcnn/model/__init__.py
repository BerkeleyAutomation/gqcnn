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

Factory functions to obtain `GQCNN`/`FCGQCNN` class based on backend.
Author: Vishal Satish
"""
from .tf import GQCNNTF, FCGQCNNTF

from autolab_core import Logger


def get_gqcnn_model(backend="tf", verbose=True):
    """Get the GQ-CNN model for the provided backend.

    Note:
        Currently only TensorFlow is supported.

    Parameters
    ----------
    backend : str
        The backend to use, currently only "tf" is supported.
    verbose : bool
        Whether or not to log initialization output to `stdout`.

    Returns
    -------
    :obj:`gqcnn.model.tf.GQCNNTF`
        GQ-CNN model with TensorFlow backend.
    """

    # Set up logger.
    logger = Logger.get_logger("GQCNNModelFactory", silence=(not verbose))

    # Return desired GQ-CNN instance based on backend.
    if backend == "tf":
        logger.info("Initializing GQ-CNN with Tensorflow as backend...")
        return GQCNNTF
    else:
        raise ValueError("Invalid backend: {}".format(backend))


def get_fc_gqcnn_model(backend="tf", verbose=True):
    """Get the FC-GQ-CNN model for the provided backend.

    Note:
        Currently only TensorFlow is supported.

    Parameters
    ----------
    backend : str
        The backend to use, currently only "tf" is supported.
    verbose : bool
        Whether or not to log initialization output to `stdout`.

    Returns
    -------
    :obj:`gqcnn.model.tf.FCGQCNNTF`
        FC-GQ-CNN model with TensorFlow backend.
    """

    # Set up logger.
    logger = Logger.get_logger("FCGQCNNModelFactory", silence=(not verbose))

    # Return desired Fully-Convolutional GQ-CNN instance based on backend.
    if backend == "tf":
        logger.info("Initializing FC-GQ-CNN with Tensorflow as backend...")
        return FCGQCNNTF
    else:
        raise ValueError("Invalid backend: {}".format(backend))
