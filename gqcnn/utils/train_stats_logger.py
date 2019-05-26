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

Handles logging of various training statistics.

Author
------
Vishal Satish
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from .enums import GQCNNFilenames


class TrainStatsLogger(object):
    """Logger for training statistics."""

    def __init__(self, experiment_dir):
        """
        Parameters
        ----------
        experiment_dir : str
            The experiment directory to save statistics to.
        """
        self.experiment_dir = experiment_dir
        self.train_eval_iters = []
        self.train_losses = []
        self.train_errors = []
        self.total_train_errors = []
        self.total_train_losses = []
        self.val_eval_iters = []
        self.val_losses = []
        self.val_errors = []
        self.val_losses = []
        self.learning_rates = []

    def log(self):
        """Flush all of the statistics to the given experiment directory."""
        np.save(os.path.join(self.experiment_dir, GQCNNFilenames.TRAIN_ITERS),
                self.train_eval_iters)
        np.save(os.path.join(self.experiment_dir, GQCNNFilenames.TRAIN_LOSSES),
                self.train_losses)
        np.save(os.path.join(self.experiment_dir, GQCNNFilenames.TRAIN_ERRORS),
                self.train_errors)
        np.save(
            os.path.join(self.experiment_dir,
                         GQCNNFilenames.TOTAL_TRAIN_ERRORS),
            self.total_train_errors)
        np.save(
            os.path.join(self.experiment_dir,
                         GQCNNFilenames.TOTAL_TRAIN_LOSSES),
            self.total_train_losses)
        np.save(os.path.join(self.experiment_dir, GQCNNFilenames.VAL_ITERS),
                self.val_eval_iters)
        np.save(os.path.join(self.experiment_dir, GQCNNFilenames.VAL_LOSSES),
                self.val_losses)
        np.save(os.path.join(self.experiment_dir, GQCNNFilenames.VAL_ERRORS),
                self.val_errors)
        np.save(
            os.path.join(self.experiment_dir, GQCNNFilenames.LEARNING_RATES),
            self.learning_rates)

    def update(self, **stats):
        """Update training statistics.

        Note
        ----
        Any statistic that is `None` in the argument dict will not be updated.

        Parameters
        ----------
        stats : dict
                Dict of statistics to be updated.
        """
        for stat, val in stats.items():
            if stat == "train_eval_iter":
                if val is not None:
                    self.train_eval_iters.append(val)
            elif stat == "train_loss":
                if val is not None:
                    self.train_losses.append(val)
            elif stat == "train_error":
                if val is not None:
                    self.train_errors.append(val)
            elif stat == "total_train_error":
                if val is not None:
                    self.total_train_errors.append(val)
            elif stat == "total_train_loss":
                if val is not None:
                    self.total_train_losses.append(val)
            elif stat == "val_eval_iter":
                if val is not None:
                    self.val_eval_iters.append(val)
            elif stat == "val_loss":
                if val is not None:
                    self.val_losses.append(val)
            elif stat == "val_error":
                if val is not None:
                    self.val_errors.append(val)
            elif stat == "learning_rate":
                if val is not None:
                    self.learning_rates.append(val)
