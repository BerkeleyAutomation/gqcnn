from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from autolab_core import YamlConfig

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler

from gqcnn.training.tf.trainer_tf_tune import GQCNNTrainerTFTune


if __name__ == "__main__":
    ray.init(local_mode=True)
    # sched = AsyncHyperBandScheduler(
    #     time_attr="timesteps_total",
    #     reward_attr="mean_accuracy",
    #     max_t=400,
    #     grace_period=20)
    config = dict(YamlConfig("cfg/train_example_pj.yaml").config)
    config["base_lr"] = tune.sample_from(
            lambda spec: np.random.uniform(0.001, 0.1))

    tune.run(
        GQCNNTrainerTFTune,
        name="exp",
        scheduler=None,
        **{
            "stop": {
                "mean_accuracy": 0.99,
                "timesteps_total": 100,
            },
            "num_samples": 1,
            "resources_per_trial": {
                "cpu": 1,
                "gpu": 0
            },
            "config": config
        })
