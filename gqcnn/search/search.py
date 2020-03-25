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

Perform hyper-parameter search over a set of GQ-CNN model/training
parameters. Actively monitor system resources and appropriately schedule
trials.

Author
------
Vishal Satish
"""
import os
import time

from .resource_manager import ResourceManager
from .trial import (GQCNNTrainingAndAnalysisTrial,
                    GQCNNFineTuningAndAnalysisTrial)
from .utils import gen_trial_params, gen_timestamp, log_trial_status
from .enums import TrialConstants, SearchConstants

from autolab_core import Logger

from ..utils import is_py2, GQCNNTrainingStatus

if is_py2():
    from Queue import Queue
else:
    from queue import Queue


class GQCNNSearch(object):
    def __init__(self,
                 analysis_config,
                 train_configs,
                 datasets,
                 split_names,
                 base_models=[],
                 output_dir=None,
                 search_name=None,
                 monitor_cpu=True,
                 monitor_gpu=True,
                 cpu_cores=[],
                 gpu_devices=[]):
        self._analysis_cfg = analysis_config

        # Create trial output dir if not specified.
        if search_name is None:
            search_name = "gqcnn_hyperparam_search_{}".format(gen_timestamp())
        if output_dir is None:
            output_dir = "models"
        self._trial_output_dir = os.path.join(output_dir, search_name)
        if not os.path.exists(self._trial_output_dir):
            os.makedirs(self._trial_output_dir)

        # Set up logger.
        self._logger = Logger.get_logger(self.__class__.__name__,
                                         log_file=os.path.join(
                                             self._trial_output_dir,
                                             "search.log"),
                                         global_log_file=True)

        # Init resource manager.
        self._resource_manager = ResourceManager(TrialConstants.TRIAL_CPU_LOAD,
                                                 TrialConstants.TRIAL_GPU_LOAD,
                                                 TrialConstants.TRIAL_GPU_MEM,
                                                 monitor_cpu=monitor_cpu,
                                                 monitor_gpu=monitor_gpu,
                                                 cpu_cores=cpu_cores,
                                                 gpu_devices=gpu_devices)

        # Parse train configs and generate individual trial parameters.
        if len(base_models) > 0:
            inconsistent_inputs_msg = ("Must have equal number of training"
                                       " configs, datasets, split_names, and"
                                       " base models!")
            assert len(train_configs) == len(datasets) == len(
                split_names) == len(base_models), inconsistent_inputs_msg
        else:
            inconsistent_inputs_msg = ("Must have equal number of training"
                                       " configs, datasets, and split_names!")
            assert len(train_configs) == len(datasets) == len(
                split_names), inconsistent_inputs_msg
        self._logger.info("Generating trial parameters...")
        trial_params = gen_trial_params(train_configs,
                                        datasets,
                                        split_names,
                                        base_models=base_models)

        # Create pending trial queue.
        self._trials_pending_queue = Queue()
        if len(base_models) > 0:
            for trial_name, hyperparam_summary, train_cfg, dataset, \
                    base_model, split_name in trial_params:
                self._trials_pending_queue.put(
                    GQCNNFineTuningAndAnalysisTrial(self._analysis_cfg,
                                                    train_cfg, dataset,
                                                    base_model, split_name,
                                                    self._trial_output_dir,
                                                    trial_name,
                                                    hyperparam_summary))
        else:
            for trial_name, hyperparam_summary, train_cfg, dataset, \
                    split_name in trial_params:
                self._trials_pending_queue.put(
                    GQCNNTrainingAndAnalysisTrial(self._analysis_cfg,
                                                  train_cfg, dataset,
                                                  split_name,
                                                  self._trial_output_dir,
                                                  trial_name,
                                                  hyperparam_summary))

        # Create containers to hold running, finished, and errored-out trials.
        self._trials_running = []
        self._trials_finished = []
        self._trials_errored = []

    def search(self):
        self._logger.info("Beginning hyper-parameter search...")
        done = False
        waiting_for_trial_init = False
        last_schedule_attempt_time = -1
        search_start_time = time.time()
        while not done:
            num_trials_pending = self._trials_pending_queue.qsize()
            num_trials_running = len(self._trials_running)
            num_trials_finished = len(self._trials_finished)
            num_trials_errored = len(self._trials_errored)

            self._logger.info(
                "----------------------------------------------------")
            self._logger.info(
                "Num trials pending: {}".format(num_trials_pending))
            self._logger.info(
                "Num trials running: {}".format(num_trials_running))
            self._logger.info(
                "Num trials finished: {}".format(num_trials_finished))
            if num_trials_errored > 0:
                self._logger.info(
                    "Num trials errored: {}".format(num_trials_errored))

            if num_trials_pending > 0 and not waiting_for_trial_init and (
                    time.time() - last_schedule_attempt_time
            ) > SearchConstants.MIN_TIME_BETWEEN_SCHEDULE_ATTEMPTS:
                self._logger.info("Attempting to schedule more trials...")
                num_trials_to_schedule, gpus_avail = \
                    self._resource_manager.num_trials_to_schedule(
                        num_trials_pending)
                self._logger.info(
                    "Scheduling {} trials".format(num_trials_to_schedule))

                if num_trials_to_schedule > 0:
                    # Start trials.
                    for _, gpu in zip(range(num_trials_to_schedule),
                                      gpus_avail):
                        trial = self._trials_pending_queue.get()
                        trial.begin(
                            gpu_avail=gpu,
                            cpu_cores_avail=self._resource_manager.cpu_cores)
                        self._trials_running.append(trial)

                    # Block scheduling until trials have started training (this
                    # is when we know what resources are still available).
                    waiting_for_trial_init = True
                last_schedule_attempt_time = time.time()

            # Check if trials have started training.
            if waiting_for_trial_init:
                training_has_started = [
                    trial.training_status == GQCNNTrainingStatus.TRAINING
                    for trial in self._trials_running
                ]
                if all(training_has_started):
                    waiting_for_trial_init = False

            # Log trial status.
            if len(self._trials_running) > 0:
                self._logger.info(log_trial_status(self._trials_running))

            # Check if any trials have finished running or errored-out.
            finished_trials_to_move = []
            errored_trials_to_move = []
            for trial in self._trials_running:
                if trial.finished:
                    finished_trials_to_move.append(trial)
                elif trial.errored_out:
                    errored_trials_to_move.append(trial)
            self._trials_finished.extend(finished_trials_to_move)
            self._trials_errored.extend(errored_trials_to_move)
            for trial in finished_trials_to_move:
                self._trials_running.remove(trial)
            for trial in errored_trials_to_move:
                self._trials_running.remove(trial)

            # Update stopping criteria and sleep.
            done = (num_trials_pending == 0) and (num_trials_running == 0)
            time.sleep(SearchConstants.SEARCH_THREAD_SLEEP)

        self._logger.info(
            "------------------Successful Trials------------------")
        self._logger.info(log_trial_status(self._trials_finished))
        if len(self._trials_errored) > 0:
            self._logger.info(
                "--------------------Failed Trials--------------------")
            self._logger.info(log_trial_status(self._trials_errored))

        self._logger.info(
            "Hyper-parameter search finished in {} seconds.".format(
                time.time() - search_start_time))
