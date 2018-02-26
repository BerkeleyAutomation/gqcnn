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
Handles logging of various optimization statistics such as error_rates/parameters/losses/etc.
Author: Vishal Satish
"""
import numpy as np
import os

class TrainStatsLogger(object):
	""" Class to log optimization error rates/parameters/losses/etc. """

	def __init__(self, experiment_dir):
		"""
        Parameters
        ----------
        experiment_dir : str
            the experiment directory to save statistics to
        """
		self.experiment_dir = experiment_dir
		self.train_eval_iters = []
		self.train_losses = []
		self.train_errors = []
		self.total_train_errors = []
		self.val_eval_iters = []
		self.val_errors = []
		self.learning_rates = []

	def log(self):
		""" Log all of the statistics to experiment directory """
		np.save(os.path.join(self.experiment_dir, 'train_eval_iters.npy'), self.train_eval_iters)
		np.save(os.path.join(self.experiment_dir, 'train_losses.npy'), self.train_losses)
		np.save(os.path.join(self.experiment_dir, 'train_errors.npy'), self.train_errors)
		np.save(os.path.join(self.experiment_dir, 'total_train_errors.npy'), self.total_train_errors)
		np.save(os.path.join(self.experiment_dir, 'val_eval_iters.npy'), self.val_eval_iters)
		np.save(os.path.join(self.experiment_dir, 'val_errors.npy'), self.val_errors)
		np.save(os.path.join(self.experiment_dir, 'learning_rates.npy'), self.learning_rates)

	def update(self, **stats):
		""" Update optimization statistics
		NOTE: Any statistic that is None in the argument dict will not be updated

		Parameters
		----------
		stats : dict
			dict of statistics and values to be updated

		"""
		for statistic in stats:
			if statistic == "train_eval_iter":
				if stats[statistic] is not None:
					self.train_eval_iters.append(stats[statistic])
			elif statistic == "train_loss":
				if stats[statistic] is not None:
					self.train_losses.append(stats[statistic])	
			elif statistic == "train_error":
				if stats[statistic] is not None:
					self.train_errors.append(stats[statistic])
			elif statistic == "total_train_error":
				if stats[statistic] is not None:
					self.total_train_errors.append(stats[statistic])
			elif statistic == "val_eval_iter":
				if stats[statistic] is not None:
					self.val_eval_iters.append(stats[statistic])
			elif statistic == "val_error":
				if stats[statistic] is not None:
					self.val_errors.append(stats[statistic])
			elif statistic == "learning_rate":
				if stats[statistic] is not None:
					self.learning_rates.append(stats[statistic])
