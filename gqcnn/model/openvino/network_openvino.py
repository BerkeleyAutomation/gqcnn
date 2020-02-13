# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 Intel Corporation. All Rights Reserved.

GQ-CNN inference with OpenVINO.

Author
------
Sharron LIU
"""
from collections import OrderedDict
import json
import math
import os
import time
import numpy as np

from autolab_core import Logger
from ...utils import (InputDepthMode, GQCNNFilenames)
from ..tf import GQCNNTF
from openvino.inference_engine import IENetwork, IECore


class GQCNNOpenVINO(GQCNNTF):
    """GQ-CNN network implemented in OpenVINO."""

    BatchSize = 64

    def __init__(self, gqcnn_config, verbose=True, log_file=None):
        """
        Parameters
        ----------
        gqcnn_config : dict
            Python dictionary of model configuration parameters.
        verbose : bool
            Whether or not to log model output to `stdout`.
        log_file : str
            If provided, model output will also be logged to this file.
        """
        self._sess = None
        # Set up logger.
        self._logger = Logger.get_logger(self.__class__.__name__,
                                         log_file=log_file,
                                         silence=(not verbose),
                                         global_log_file=verbose)
        self._parse_config(gqcnn_config)

    @staticmethod
    def load(model_dir, device, verbose=True, log_file=None):
        """Instantiate a trained GQ-CNN for fine-tuning or inference.

        Parameters
        ----------
        model_dir : str
            Path to trained GQ-CNN model.
        device : str
            Device type for inference to execute CPU|GPU|MYRIAD
        verbose : bool
            Whether or not to log model output to `stdout`.
        log_file : str
            If provided, model output will also be logged to this file.

        Returns
        -------
        :obj:`GQCNNOpenVINO`
            Initialized GQ-CNN.
        """
        # Load GQCNN config
        config_file = os.path.join(model_dir, GQCNNFilenames.SAVED_CFG)
        with open(config_file) as data_file:
            train_config = json.load(data_file, object_pairs_hook=OrderedDict)
        # Support for legacy configs.
        try:
            gqcnn_config = train_config["gqcnn"]
        except KeyError:
            gqcnn_config = train_config["gqcnn_config"]
            gqcnn_config["debug"] = 0
            gqcnn_config["seed"] = 0
            # Legacy networks had no angular support.
            gqcnn_config["num_angular_bins"] = 0
            # Legacy networks only supported depth integration through pose
            # stream.
            gqcnn_config["input_depth_mode"] = InputDepthMode.POSE_STREAM

        # Initialize OpenVINO network
        gqcnn = GQCNNOpenVINO(gqcnn_config, verbose=verbose, log_file=log_file)
        if (device == "MYRIAD"):  # MYRIAD batch size force to 1
            gqcnn.set_batch_size(1)
        else:
            gqcnn.set_batch_size(64)

        # Initialize input tensors for prediction
        gqcnn._input_im_arr = np.zeros((gqcnn._batch_size, gqcnn._im_height,
                                        gqcnn._im_width, gqcnn._num_channels))
        gqcnn._input_pose_arr = np.zeros((gqcnn._batch_size, gqcnn._pose_dim))

        # Initialize mean tensor and standard tensor
        gqcnn.init_mean_and_std(model_dir)

        # Load OpenVINO network on specified device
        gqcnn.load_openvino(model_dir, device)

        return gqcnn

    def open_session(self):
        pass

    def close_session(self):
        pass

    def load_openvino(self, model_dir, device):
        self._ie = IECore()
        # load OpenVINO executable network to device
        model_path = os.path.split(model_dir)
        model_xml = os.path.join(model_path[0], "OpenVINO", model_path[1])
        model_xml = os.path.join(model_xml, "FP16",
                                 "inference_graph_frozen.xml")
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self._vino_net = IENetwork(model_xml, model_bin)
        self._vino_net.batch_size = self._batch_size
        assert len(self._vino_net.inputs.keys()) == 2, "GQCNN two input nodes"
        assert len(self._vino_net.outputs) == 1, "GQCNN one output node"
        vino_inputs = iter(self._vino_net.inputs)
        self._input_im = next(vino_inputs)
        self._input_pose = next(vino_inputs)
        self._output_blob = next(iter(self._vino_net.outputs))
        self._vino_exec_net = self._ie.load_network(network=self._vino_net,
                                                    device_name=device)

    def unload_openvino(self):
        del self._vino_exec_net
        del self._vino_net
        del self._ie

    def predict_openvino(self, image_arr, pose_arr, verbose=False):
        """ Predict a set of images in batches
        Parameters
        ----------
        image_arr : :obj:`tensorflow Tensor`
            4D Tensor of images to be predicted
        pose_arr : :obj:`tensorflow Tensor`
            4D Tensor of poses to be predicted
        """

        # Get prediction start time.
        start_time = time.time()

        if verbose:
            self._logger.info("Predicting...")

        # Setup for prediction.
        num_batches = math.ceil(image_arr.shape[0] / self._batch_size)
        num_images = image_arr.shape[0]
        num_poses = pose_arr.shape[0]

        output_arr = np.zeros(
            [num_images] +
            list(self._vino_net.outputs[self._output_blob].shape[1:]))
        if num_images != num_poses:
            raise ValueError("Must provide same number of images as poses!")

        # Predict in batches.
        i = 0
        batch_idx = 0
        while i < num_images:
            if verbose:
                self._logger.info("Predicting batch {} of {}...{}".format(
                    batch_idx, num_batches, num_images))
            batch_idx += 1
            dim = min(self._batch_size, num_images - i)
            cur_ind = i
            end_ind = cur_ind + dim

            if self._input_depth_mode == InputDepthMode.POSE_STREAM:
                self._input_im_arr[:dim, ...] = (
                    image_arr[cur_ind:end_ind, ...] -
                    self._im_mean) / self._im_std
                self._input_pose_arr[:dim, :] = (
                    pose_arr[cur_ind:end_ind, :] -
                    self._pose_mean) / self._pose_std
            elif self._input_depth_mode == InputDepthMode.SUB:
                self._input_im_arr[:dim, ...] = image_arr[cur_ind:end_ind, ...]
                self._input_pose_arr[:dim, :] = pose_arr[cur_ind:end_ind, :]
            elif self._input_depth_mode == InputDepthMode.IM_ONLY:
                self._input_im_arr[:dim, ...] = (
                    image_arr[cur_ind:end_ind, ...] -
                    self._im_mean) / self._im_std

            n, c, h, w = self._vino_net.inputs[self._input_im].shape
            input_im_arr = self._input_im_arr.reshape((n, c, h, w))
            res = self._vino_exec_net.infer(
                inputs={
                    self._input_im: input_im_arr,
                    self._input_pose: self._input_pose_arr
                })

            # Allocate output tensor.
            output_arr[cur_ind:end_ind, :] = res[self._output_blob][:dim, :]
            i = end_ind

        # Get total prediction time.
        pred_time = time.time() - start_time
        if verbose:
            self._logger.info("Prediction took {} seconds.".format(pred_time))

        return output_arr

    def predict(self, image_arr, pose_arr, verbose=False):
        """Predict the probability of grasp success given a depth image and
        gripper pose.

        Parameters
        ----------
        image_arr : :obj:`numpy ndarray`
            4D tensor of depth images.
        pose_arr : :obj:`numpy ndarray`
            Tensor of gripper poses.
        verbose : bool
            Whether or not to log progress to stdout, useful to turn off during
            training.
        """
        return self.predict_openvino(image_arr, pose_arr, verbose=verbose)
