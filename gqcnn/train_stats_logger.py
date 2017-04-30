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
