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
FC-GQ-CNN network implemented in Tensorflow for use with the Intel Neural Compute Stick (NCS) and NCSDK v2.
Author: Vishal Satish
"""
import os
import json
from collections import OrderedDict
import subprocess
import time
import math

import tensorflow as tf
from mvnc import mvncapi as mvnc
import numpy as np

from fc_network_tf import FCGQCNNTF
from gqcnn.utils import InputDepthMode, TrainingMode, NCSFCGQCNNTFFileNames, NCSFCGQCNNTFNodes

def pairwise_softmax(tensor):
    """ Applies pair-wise softmax to dim -1 of tensor. """
    tensor_soft = np.copy(tensor)
    for i in xrange(0, tensor_soft.shape[-1], 2):
        tensor_soft[..., i:i+2] = np.exp(tensor_soft[..., i:i+2]) / np.sum(np.exp(tensor_soft[..., i:i+2]), axis=-1, keepdims=True) 
    return tensor_soft

class NCSFCGQCNNTF(FCGQCNNTF):
    """FC-GQ-CNN network implemented in Tensorflow for use with the Intel Neural Compute Stick (NCS) and NCSDK v2. Key implementation details: 
        1) Only uses Tensorflow operations supported by the NCSDK. These can be found in /usr/local/bin/ncsdk/Controllers/Parsers/TensorFlowParser/. For the FC-GQ-CNN, the pair-wise softmax layer depends on the unsupported tf.split() operation. We instead implement it with iterative tf.slice operations.
        2) The save() function saves a clean protobuf (.pb) version of the network without training operations that can be used for compilation with the NCSDK.
        3) Sets the batch size to 1 as required by the NCS.
    """

    @staticmethod
    def load(model_dir, fc_config, compiler_extra_args={}, log_file=None):
        """Instantiate an FC-GQ-CNN for use with the Intel NCS from a trained GQ-CNN. 

        Parameters
        ----------
        model_dir : str
            path to trained GQ-CNN

        Returns
        -------
        :obj:`NCSFCGQCNNTF`
            initialized NCSFCGQCNNTF 
        """
        config_file = os.path.join(model_dir, 'config.json')
        with open(config_file) as data_file:    
            train_config = json.load(data_file, object_pairs_hook=OrderedDict)
        gqcnn_config = train_config['gqcnn']

        fcgqcnn = NCSFCGQCNNTF(gqcnn_config, fc_config, log_file=log_file)

        assert fcgqcnn._input_depth_mode in [InputDepthMode.IM_ONLY, InputDepthMode.SUB], 'Pose stream is not supported with NCS!'

        # load the weights and normalization metrics
        fcgqcnn.init_weights_file(os.path.join(model_dir, 'model.ckpt'))
        fcgqcnn.init_mean_and_std(model_dir)

        # the NCSDK requires a bsz of 1
        fcgqcnn.set_batch_size(1)

        # set the model dir because we may want to save things like the cleaned model 
        fcgqcnn.set_model_dir(model_dir)

        # initialize the network
        training_mode = train_config['training_mode']
        if training_mode == TrainingMode.CLASSIFICATION:
            fcgqcnn.initialize_network(add_softmax=True)
        elif training_mode == TrainingMode.REGRESSION:
            fcgqcnn.initialize_network()
        else:
            raise ValueError('Invalid training mode: {}'.format(training_mode))

        # compile the network for inference on the NCS
        fcgqcnn.set_compiler_extra_args(compiler_extra_args)
        fcgqcnn.compile()

        return fcgqcnn

    def set_compiler_extra_args(self, args):
        self._compiler_extra_args = args
    
    def add_softmax_to_output(self, num_outputs=0):
        """Adds softmax to output of network. Uses iterative tf.slice operation instead of tf.split in pair-wise softmax implementation."""
        with tf.name_scope('softmax'):
            if num_outputs > 0: 
                self._logger.info('Building Pair-wise Softmax Layer...')
                binwise_split_output = [tf.slice(self._output_tensor, (0, 0, 0, i*2), self._output_tensor.get_shape().as_list()[:-1] + [2]) for i in range(num_outputs)]
                binwise_split_output_soft = [tf.nn.softmax(s, name='output_%03d'%(i)) for i, s in enumerate(binwise_split_output)]
                self._output_tensor = tf.concat(binwise_split_output_soft, -1, name='output')
            else:    
                self._logger.info('Building Softmax Layer...')
                self._output_tensor = tf.nn.softmax(self._output_tensor, name='output')

    def _prepare_device(self):
        """Prepare the NCS for inference."""

        self._logger.info('Initializing NCS device...')
        device = mvnc.Device(mvnc.enumerate_devices()[0])
        device.open()

        self._logger.info('Loading graph onto device...')
        with open(os.path.join(self._model_dir, NCSFCGQCNNTFFileNames.NCS_GRAPH)) as fhandle:
            g_buff = fhandle.read()
        graph = mvnc.Graph('fcgqcnn_graph')
        return device, graph, graph.allocate_with_fifos(device, g_buff)

    def _teardown_device(self):
        """Close and clean-up the NCS."""

        self._logger.info('Tearing down NCS...')
        self._fifo_in.destroy()
        self.fifo_out.destroy()
        self._ncs_graph.destroy()
        self._device.close()
     
    def _predict(self, image_arr, pose_arr, verbose=False):
        """Query network predictions with NCS.

        Parameters
        ----------
        image_arr :obj:`numpy.ndarray`
            input images
        pose_arr :obj:`numpy.ndarray`
            input gripper poses
        verbose : bool
            whether or not to log progress, useful to turn off during training
        """       
        # get prediction start time
        start_time = time.time()

        if verbose:
            self._logger.info('Predicting...')

        # setup for prediction
        num_batches = math.ceil(image_arr.shape[0] / self._batch_size)
        num_images = image_arr.shape[0]
        num_poses = pose_arr.shape[0]

        output_arr = None
        if num_images != num_poses:
            raise ValueError('Must provide same number of images as poses!')

        # initialize NCS
        self._device, self._ncs_graph, (self._fifo_in, self._fifo_out) = self._prepare_device()

        # predict in batches
        i = 0
        batch_idx = 0
        while i < num_images:
            if verbose:
                self._logger.info('Predicting batch {} of {}...'.format(batch_idx, num_batches))
            batch_idx += 1
            dim = min(self._batch_size, num_images - i)
            cur_ind = i
            end_ind = cur_ind + dim

            # subtract the depth and then normalize
            if self._input_depth_mode == InputDepthMode.SUB:
                # read batch
                images = image_arr[cur_ind:end_ind, ...]
                if len(pose_arr.shape) == 1:
                    poses = pose_arr[cur_ind:end_ind]
                else:
                    poses = pose_arr[cur_ind:end_ind, :]

                # subtract poses
                images_sub = images - np.tile(np.reshape(poses, (-1, 1, 1, 1)), (1, images.shape[1], images.shape[2], 1))

                # normalize
                self._input_im_arr[:dim, ...] = (images_sub - self._im_depth_sub_mean) / self._im_depth_sub_std

            # normalize the images
            elif self._input_depth_mode == InputDepthMode.IM_ONLY:
                self._input_im_arr[:dim, ...] = (
                    image_arr[cur_ind:end_ind, ...] - self._im_mean) / self._im_std

            # convert inputs to 32-bit for inference
            input_im_arr_32 = self._input_im_arr.astype(np.float32)

            # run forward inference
            self._ncs_graph.queue_inference_with_fifo_elem(self._fifo_in, self._fifo_out, input_im_arr_32, None)
            gqcnn_output, _ = self._fifo_out.read_elem()

            # reshape 1d output from queue
            gqcnn_output = gqcnn_output.reshape((self._batch_size,) + self._fc_output_shape)
            
            # manually apply pair-wise softmax outside of graph
            gqcnn_output = pairwise_softmax(gqcnn_output)

            # allocate output tensor
            if output_arr is None:
                output_arr = np.zeros([num_images] + list(gqcnn_output.shape[1:]))

            output_arr[cur_ind:end_ind, :] = gqcnn_output[:dim, :]
            i = end_ind
    
        # get total prediction time
        pred_time = time.time() - start_time
        if verbose:
            self._logger.info('Prediction took {} seconds.'.format(pred_time))

        return output_arr

    def _save_proto(self):
        """Generates a clean protobuf (.pb) version of the network without training operations."""

        save_fname = NCSFCGQCNNTFFileNames.TF_PROTO
        if save_fname in os.listdir(self._model_dir):
            valid_responses = ['y', 'n']
            user_in = raw_input('Found existing {} in model_dir. Overwrite? (Y/N): '.format(save_fname)).lower()
            while user_in not in valid_responses:
                user_in = raw_input('Invalid response "{}", please enter either (Y)es or (N)o: '.format(user_in))
            if user_in == 'y':
                self._logger.info('Overwriting existing {}...'.format(save_fname))
            else:
                return

        self._logger.info('Saving clean network for inference...')
        self._logger.info('Building cleaned graph...')
        self.open_session()
        cleaned_graph = tf.graph_util.convert_variables_to_constants(self._sess, self._graph.as_graph_def(), [NCSFCGQCNNTFNodes.OUT])
        self._logger.info('Writing clean graph...')
        with tf.gfile.GFile(os.path.join(self._model_dir, save_fname), 'wb') as f:
            f.write(cleaned_graph.SerializeToString())
        self.close_session()

    def compile(self):
        """Compiles the network for inference on the NCS. Generates a (.graph) file."""

        # first save a clean model protobuf for input to the compiler
        self._save_proto()

        # compile
        save_fname = NCSFCGQCNNTFFileNames.NCS_GRAPH
        if save_fname in os.listdir(self._model_dir):
            valid_responses = ['y', 'n']
            user_in = raw_input('Found existing {} in model_dir. Overwrite? (Y/N): '.format(save_fname)).lower()
            while user_in not in valid_responses:
                user_in = raw_input('Invalid response "{}", please enter either (Y)es or (N)o: '.format(user_in))
            if user_in == 'y':
                self._logger.info('Overwriting existing {}...'.format(save_fname))
            else:
                return

        self._logger.info('Compiling model for NCS...')
        command = ['mvNCCompile', os.path.join(self._model_dir, NCSFCGQCNNTFFileNames.TF_PROTO), '-in', NCSFCGQCNNTFNodes.IN, '-on', NCSFCGQCNNTFNodes.OUT, '-o', os.path.join(self._model_dir, save_fname)]
        for arg, val in self._compiler_extra_args.iteritems():
            command.append(arg)
            command.append(str(val))
        self._logger.info('Executing: "{}"'.format(' '.join(command)))
        p = subprocess.Popen(command)
        p.wait()
