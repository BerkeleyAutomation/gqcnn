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

Intelligent resource manager for hyper-parameter search. Queries resources
available and appropriately distributes resources over possible trials to run.

Author
------
Vishal Satish
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import time

import GPUtil
import numpy as np
import psutil

from autolab_core import Logger

CPU_LOAD_SAMPLE_INTERVAL = 4.0
# This is a hack because it seems that psutil is returning a lower load than
# htop, which could be because htop takes into account queued tasks.
CPU_LOAD_OFFSET = 50
GPU_STAT_NUM_SAMPLES = 4
GPU_STAT_SAMPLE_INTERVAL = 1.0


class ResourceManager(object):
    def __init__(self,
                 trial_cpu_load,
                 trial_gpu_load,
                 trial_gpu_mem,
                 monitor_cpu=True,
                 monitor_gpu=True,
                 cpu_cores=[],
                 gpu_devices=[]):
        self._monitor_cpu = monitor_cpu
        self._monitor_gpu = monitor_gpu

        # Set up logger.
        self._logger = Logger.get_logger(self.__class__.__name__)

        if not monitor_cpu:
            self._logger.warning(
                "Not monitoring cpu resources is not advised.")
        if not monitor_gpu:
            self._logger.warning(
                "Not monitoring gpu resources is not advised.")

        self._trial_cpu_load = trial_cpu_load
        self._trial_gpu_load = trial_gpu_load
        self._trial_gpu_mem = trial_gpu_mem

        self._cpu_cores = cpu_cores
        if len(self._cpu_cores) == 0:
            self._logger.warning(
                "No CPU cores specified-proceeding to use all available cores."
            )
            self._cpu_cores = range(psutil.cpu_count())
        self._cpu_count = len(self._cpu_cores)

        self._gpu_devices = gpu_devices
        if len(self._gpu_devices) == 0:
            no_gpus_specified_msg = ("No GPU devices specified-proceeding to"
                                     " use all available devices.")
            self._logger.warning(no_gpus_specified_msg)
            self._gpu_devices = range(len(GPUtil.getGPUs()))

    @property
    def cpu_cores(self):
        return self._cpu_cores

    def _get_cpu_load(self):
        self._logger.info("Sampling cpu load...")
        cpu_core_loads = psutil.cpu_percent(interval=CPU_LOAD_SAMPLE_INTERVAL,
                                            percpu=True)
        total_load = 0
        for core in self._cpu_cores:
            total_load += cpu_core_loads[core]
        return total_load + CPU_LOAD_OFFSET

    def _get_gpu_stats(self):
        self._logger.info("Sampling gpu memory and load...")
        gpu_samples = []
        for _ in range(GPU_STAT_NUM_SAMPLES):
            gpu_samples.append(GPUtil.getGPUs())
            time.sleep(GPU_STAT_SAMPLE_INTERVAL)
        num_gpus = len(gpu_samples[0])
        sample_loads = np.zeros((num_gpus, GPU_STAT_NUM_SAMPLES))
        sample_mems = np.zeros((num_gpus, GPU_STAT_NUM_SAMPLES))
        total_mems = np.zeros((num_gpus, ))
        for i in range(GPU_STAT_NUM_SAMPLES):
            for gpu in gpu_samples[i]:
                if gpu.id in self._gpu_devices:
                    sample_loads[gpu.id, i] = gpu.load * 100
                    sample_mems[gpu.id, i] = gpu.memoryUsed
                else:
                    # Trick the manager into thinking the GPU is fully utilized
                    # so it will never be chosen.
                    sample_loads[gpu.id, i] = 100
                    sample_mems[gpu.id, i] = gpu.memoryTotal
                total_mems[gpu.id] = gpu.memoryTotal
        return total_mems.tolist(), np.mean(
            sample_loads, axis=1).tolist(), np.mean(sample_mems,
                                                    axis=1).tolist()

    def _build_gpu_list(self, max_possible_trials_per_device):
        gpus_avail = []
        for device_id, max_trials in enumerate(max_possible_trials_per_device):
            for _ in range(max_trials):
                gpus_avail.append(str(device_id))
        # This is because we might truncate this list later because of a more
        # severe resource bottleneck, in which case we want to evenly
        # distribute the load.
        random.shuffle(gpus_avail)
        return gpus_avail

    def num_trials_to_schedule(self, num_pending_trials):
        num_trials_to_schedule = num_pending_trials
        if self._monitor_cpu:  # Check cpu bandwith.
            cpu_load = min(self._get_cpu_load(), self._cpu_count * 100)
            max_possible_trials_cpu = int(
                (self._cpu_count * 100 - cpu_load) // self._trial_cpu_load)
            self._logger.info("CPU load: {}%, Max possible trials: {}".format(
                cpu_load, max_possible_trials_cpu))
            num_trials_to_schedule = min(num_trials_to_schedule,
                                         max_possible_trials_cpu)

        if self._monitor_gpu:  # Check gpu bandwith.
            total_gpu_mems, gpu_loads, gpu_mems = self._get_gpu_stats()
            max_possible_trials_gpu_load_per_device = [
                int((100 - gpu_load) // self._trial_gpu_load)
                for gpu_load in gpu_loads
            ]
            max_possible_trials_gpu_mem_per_device = [
                int((total_gpu_mem - gpu_mem) // self._trial_gpu_mem)
                for total_gpu_mem, gpu_mem in zip(total_gpu_mems, gpu_mems)
            ]
            max_possible_trials_gpu_per_device = map(
                lambda x: min(x[0], x[1]),
                zip(max_possible_trials_gpu_load_per_device,
                    max_possible_trials_gpu_mem_per_device))
            self._logger.info(
                "GPU loads: {}, GPU mems: {}, Max possible trials: {}".format(
                    "% ".join([str(gpu_load) for gpu_load in gpu_loads]) + "%",
                    "MiB ".join([str(gpu_mem)
                                 for gpu_mem in gpu_mems]) + "MiB",
                    sum(max_possible_trials_gpu_per_device)))
            num_trials_to_schedule = min(
                num_trials_to_schedule,
                sum(max_possible_trials_gpu_per_device))

            # Build the device list for scheduling trials on specific gpus.
            gpus_avail = self._build_gpu_list(
                max_possible_trials_gpu_per_device)
        else:
            # Just distribute load among gpus.
            num_gpus = self._get_gpu_count()
            trials_per_gpu = int(math.ceil(num_trials_to_schedule / num_gpus))
            gpus_avail = self._build_gpu_list([trials_per_gpu] * num_gpus)

        gpus_avail = gpus_avail[:num_trials_to_schedule]
        self._logger.info(
            "Max possible trials overall: {}".format(num_trials_to_schedule))
        return num_trials_to_schedule, gpus_avail
