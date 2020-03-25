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

Constants/enums.

Author
------
Vishal Satish
"""
import math

import tensorflow as tf


# Other constants.
class GeneralConstants(object):
    SEED = 3472134
    SEED_SAMPLE_MAX = 2**32 - 1  # Max range for `np.random.seed`.
    timeout_option = tf.RunOptions(timeout_in_ms=1000000)
    MAX_PREFETCH_Q_SIZE = 250
    NUM_PREFETCH_Q_WORKERS = 3
    QUEUE_SLEEP = 0.001
    PI = math.pi
    FIGSIZE = 16  # For visualization.


# Enum for image modalities.
class ImageMode(object):
    BINARY = "binary"
    DEPTH = "depth"
    BINARY_TF = "binary_tf"
    COLOR_TF = "color_tf"
    GRAY_TF = "gray_tf"
    DEPTH_TF = "depth_tf"
    DEPTH_TF_TABLE = "depth_tf_table"
    TF_DEPTH_IMS = "tf_depth_ims"


# Enum for training modes.
class TrainingMode(object):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"  # Has not been tested, for experimentation only!


# Enum for input pose data formats.
class GripperMode(object):
    PARALLEL_JAW = "parallel_jaw"
    SUCTION = "suction"
    MULTI_SUCTION = "multi_suction"
    LEGACY_PARALLEL_JAW = "legacy_parallel_jaw"
    LEGACY_SUCTION = "legacy_suction"


# Enum for input depth mode.
class InputDepthMode(object):
    POSE_STREAM = "pose_stream"
    SUB = "im_depth_sub"
    IM_ONLY = "im_only"


# Enum for training status.
class GQCNNTrainingStatus(object):
    NOT_STARTED = "not_started"
    SETTING_UP = "setting_up"
    TRAINING = "training"


# Enum for filenames.
class GQCNNFilenames(object):
    PCT_POS_VAL = "pct_pos_val.npy"
    PCT_POS_TRAIN = "pct_pos_train.npy"
    LEARNING_RATES = "learning_rates.npy"

    TRAIN_ITERS = "train_eval_iters.npy"
    TRAIN_LOSSES = "train_losses.npy"
    TRAIN_ERRORS = "train_errors.npy"
    TOTAL_TRAIN_LOSSES = "total_train_losses.npy"
    TOTAL_TRAIN_ERRORS = "total_train_errors.npy"

    VAL_ITERS = "val_eval_iters.npy"
    VAL_LOSSES = "val_losses.npy"
    VAL_ERRORS = "val_errors.npy"

    LEG_MEAN = "mean.npy"
    LEG_STD = "std.npy"
    IM_MEAN = "im_mean.npy"
    IM_STD = "im_std.npy"
    IM_DEPTH_SUB_MEAN = "im_depth_sub_mean.npy"
    IM_DEPTH_SUB_STD = "im_depth_sub_std.npy"
    POSE_MEAN = "pose_mean.npy"
    POSE_STD = "pose_std.npy"

    FINAL_MODEL = "model.ckpt"
    INTER_MODEL = "model_{}.ckpt"

    SAVED_ARCH = "architecture.json"
    SAVED_CFG = "config.json"
