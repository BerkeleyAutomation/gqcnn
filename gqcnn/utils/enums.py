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
Class for storing constants/enums for optimization
Author: Vishal Satish
"""
import tensorflow as tf

# other constants
class GeneralConstants:
    SEED = 3472134
    timeout_option = tf.RunOptions(timeout_in_ms=1000000)
    JSON_INDENT = 2
    QUEUE_CAPACITY = 1000
    QUEUE_SLEEP = 0.001
    
# enum for image modalities
class ImageMode:
    BINARY = 'binary'
    DEPTH = 'depth'
    BINARY_TF = 'binary_tf'
    COLOR_TF = 'color_tf'
    GRAY_TF = 'gray_tf'
    DEPTH_TF = 'depth_tf'
    DEPTH_TF_TABLE = 'depth_tf_table'
    TF_DEPTH_IMS = 'tf_depth_ims'
    
# enum for training modes
class TrainingMode:
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression' # has not been shown to work, for experimentation only!

# enum for input data formats
class GripperMode:
    PARALLEL_JAW = 'parallel_jaw'
    SUCTION = 'suction'
    LEGACY_PARALLEL_JAW = 'legacy_parallel_jaw'
    LEGACY_SUCTION = 'legacy_suction'
