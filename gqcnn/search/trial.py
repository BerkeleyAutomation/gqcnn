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

Trials for hyper-parameter search.

Author
------
Vishal Satish
"""
from abc import ABC, abstractmethod
import json
import multiprocessing
import os
import sys

import numpy as np

from ..model import get_gqcnn_model
from ..training import get_gqcnn_trainer
from ..utils import GeneralConstants, GQCNNTrainingStatus
from ..analysis import GQCNNAnalyzer


class TrialStatus:
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    EXCEPTION = "exception"


class GQCNNTrialWithAnalysis(ABC):
    def __init__(self, analysis_cfg, train_cfg, dataset_dir, split_name,
                 output_dir, model_name, hyperparam_summary):
        self._analysis_cfg = analysis_cfg
        self._train_cfg = train_cfg
        self._dataset_dir = dataset_dir
        self._split_name = split_name
        self._output_dir = output_dir
        self._model_name = model_name
        self._hyperparam_summary = hyperparam_summary
        self._manager = multiprocessing.Manager()
        # To communicate with training.
        self._train_progress_dict = self._build_train_progress_dict()
        # To communicate with trial.
        self._trial_progress_dict = self._build_trial_progress_dict()
        self._process = None

    def _build_train_progress_dict(self):
        progress_dict = self._manager.dict(
            training_status=GQCNNTrainingStatus.NOT_STARTED,
            epoch=np.nan,
            analysis=None)
        return progress_dict

    def _build_trial_progress_dict(self):
        progress_dict = self._manager.dict(status=TrialStatus.PENDING)
        return progress_dict

    @abstractmethod
    def _run(self, trainer):
        pass

    def _run_trial(self,
                   analysis_config,
                   train_config,
                   dataset_dir,
                   split_name,
                   output_dir,
                   model_name,
                   train_progress_dict,
                   trial_progress_dict,
                   hyperparam_summary,
                   gpu_avail="",
                   cpu_cores_avail=[],
                   backend="tf"):
        trial_progress_dict["status"] = TrialStatus.RUNNING
        try:
            os.system("taskset -pc {} {}".format(
                ",".join(str(i) for i in cpu_cores_avail), os.getpid()))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_avail

            gqcnn = get_gqcnn_model(backend,
                                    verbose=False)(train_config["gqcnn"],
                                                   verbose=False)
            trainer = get_gqcnn_trainer(backend)(
                gqcnn,
                dataset_dir,
                split_name,
                output_dir,
                train_config,
                name=model_name,
                progress_dict=train_progress_dict,
                verbose=False)
            self._run(trainer)

            with open(
                    os.path.join(output_dir, model_name,
                                 "hyperparam_summary.json"), "wb") as fhandle:
                json.dump(hyperparam_summary,
                          fhandle,
                          indent=GeneralConstants.JSON_INDENT)

            train_progress_dict["training_status"] = "analyzing"
            analyzer = GQCNNAnalyzer(analysis_config, verbose=False)
            _, _, init_train_error, final_train_error, init_train_loss, \
                final_train_loss, init_val_error, final_val_error, \
                norm_final_val_error = analyzer.analyze(
                    os.path.join(output_dir, model_name), output_dir)
            analysis_dict = {}
            analysis_dict["init_train_error"] = init_train_error
            analysis_dict["final_train_error"] = final_train_error
            analysis_dict["init_train_loss"] = init_train_loss
            analysis_dict["final_train_loss"] = final_train_loss
            analysis_dict["init_val_error"] = init_val_error
            analysis_dict["final_val_error"] = final_val_error
            analysis_dict["norm_final_val_error"] = norm_final_val_error
            train_progress_dict["analysis"] = analysis_dict

            train_progress_dict["training_status"] = "finished"
            trial_progress_dict["status"] = TrialStatus.FINISHED
            sys.exit(0)
        except Exception as e:
            trial_progress_dict["status"] = TrialStatus.EXCEPTION
            trial_progress_dict["error_msg"] = str(e)
            sys.exit(0)

    @property
    def finished(self):
        return self._trial_progress_dict["status"] == TrialStatus.FINISHED

    @property
    def errored_out(self):
        return self._trial_progress_dict["status"] == TrialStatus.EXCEPTION

    @property
    def error_msg(self):
        return self._trial_progress_dict["error_msg"]

    @property
    def training_status(self):
        return self._train_progress_dict["training_status"]

    def begin(self, gpu_avail="", cpu_cores_avail=[]):
        self._status = TrialStatus.RUNNING
        self._process = multiprocessing.Process(
            target=self._run_trial,
            args=(self._analysis_cfg, self._train_cfg, self._dataset_dir,
                  self._split_name, self._output_dir, self._model_name,
                  self._train_progress_dict, self._trial_progress_dict,
                  self._hyperparam_summary, gpu_avail, cpu_cores_avail))
        self._process.start()

    def __str__(self):
        trial_str = "Trial: {}, Training Stage: {}".format(
            self._model_name, self.training_status)
        if self.training_status == GQCNNTrainingStatus.TRAINING and not \
                np.isnan(self._train_progress_dict["epoch"]):
            trial_str += ", Epoch: {}/{}".format(
                self._train_progress_dict["epoch"],
                self._train_cfg["num_epochs"])
        if self.errored_out:
            trial_str += ", Error message: {}".format(self.error_msg)
        if self.training_status == "finished":
            finished_msg = (", Initial train error: {}, Final train error: {},"
                            " Initial train loss: {}, Final train loss: {},"
                            " Initial val error: {}, Final val error: {}, Norm"
                            " final val error: {}")
            trial_str += finished_msg.format(
                self._train_progress_dict["analysis"]["init_train_error"],
                self._train_progress_dict["analysis"]["final_train_error"],
                self._train_progress_dict["analysis"]["init_train_loss"],
                self._train_progress_dict["analysis"]["final_train_loss"],
                self._train_progress_dict["analysis"]["init_val_error"],
                self._train_progress_dict["analysis"]["final_val_error"],
                self._train_progress_dict["analysis"]["norm_final_val_error"])
        return trial_str


class GQCNNTrainingAndAnalysisTrial(GQCNNTrialWithAnalysis):
    def _run(self, trainer):
        trainer.train()


class GQCNNFineTuningAndAnalysisTrial(GQCNNTrialWithAnalysis):
    def __init__(self, analysis_cfg, train_cfg, dataset_dir, base_model_dir,
                 split_name, output_dir, model_name, hyperparam_summary):
        GQCNNTrialWithAnalysis.__init__(self, analysis_cfg, train_cfg,
                                        dataset_dir, split_name, output_dir,
                                        model_name, hyperparam_summary)

        self._base_model_dir = base_model_dir

    def _run(self, trainer):
        trainer.finetune(self._base_model_dir)
