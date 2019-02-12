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
FC-GQ-CNN network implemented in Tensorflow for use with the Intel NCSDK 2
Author: Vishal Satish
"""
import os
import json
from collections import OrderedDict

import tensorflow as tf

from fc_network_tf import FCGQCNNTF
from gqcnn.utils import TrainingMode

class NCFCGQCNNTF(FCGQCNNTF):
    """FC-GQ-CNN network implemented in Tensorflow for use with the Intel NCSDK 2. Key implementation details: 
        1) Only uses Tensorflow operations supported by the NCSDK. These can be found in /usr/local/bin/ncsdk/Controllers/Parsers/TensorFlowParser/. For the FC-GQ-CNN, the pair-wise softmax layer depends on the unsupported tf.split() operation. We instead implement it with iterative tf.slice operations.
        2) The save() function saves a clean protobuf (.pb) version of the network without training operations that can be used for compilation with the NCSDK.
        3) Sets the batch size to 1 as required by the NCSDK.
    """

    @staticmethod
    def load(model_dir, fc_config, log_file=None):
        """Instantiate an FC-GQ-CNN for use with the Intel NCSDK 2 from a trained GQ-CNN. 

        Parameters
        ----------
        model_dir : str
            path to trained GQ-CNN

        Returns
        -------
        :obj:`NCFCGQCNNTF`
            initialized NCFCGQCNNTF 
        """
        config_file = os.path.join(model_dir, 'config.json')
        with open(config_file) as data_file:    
            train_config = json.load(data_file, object_pairs_hook=OrderedDict)
        gqcnn_config = train_config['gqcnn']

        fcgqcnn = NCFCGQCNNTF(gqcnn_config, fc_config, log_file=log_file)

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
        return fcgqcnn

    def add_softmax_to_output(self, num_outputs=0):
        """Adds softmax to output of network. Uses iterative tf.slice operation instead of tf.split in pair-wise softmax implementation."""
        with tf.name_scope('softmax'):
            if num_outputs > 0: 
                self._logger.info('Building Pair-wise Softmax Layer...')
                binwise_split_output = [tf.slice(self._output_tensor, (0, 0, 0, i*2), self._output_tensor.get_shape().as_list()[:-1] + [2]) for i in range(self._angular_bins)]
                binwise_split_output_soft = [tf.nn.softmax(s, name='output_%03d'%(i)) for i, s in enumerate(binwise_split_output)]
                self._output_tensor = tf.concat(binwise_split_output_soft, -1, name='output')
            else:    
                self._logger.info('Building Softmax Layer...')
                self._output_tensor = tf.nn.softmax(self._output_tensor, name='output')

    def save(self):
        """Generates a clean protobuf (.pb) version of the network without training operations."""
        self._logger.info('Saving clean network for inference...')

        self._logger.info('Building cleaned graph...')
        self.open_session()
        cleaned_graph = tf.graph_util.convert_variables_to_constants(self._sess, self._graph.as_graph_def(), ['softmax/output'])

        self._logger.info('Writing clean graph...')
        with tf.gfile.GFile(os.path.join(self._model_dir, 'model.pb'), 'wb') as f:
            f.write(cleaned_graph.SerializeToString())

        self.close_session()
