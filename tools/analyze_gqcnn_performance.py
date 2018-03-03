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
Analyzes a GQ-CNN model.

Author
------
Vishal Satish and Jeff Mahler
"""
import argparse
import logging
import os
import time
import os

from autolab_core import YamlConfig
from gqcnn import GQCNN, SGDOptimizer, GQCNNAnalyzer

if __name__ == '__main__':
    # setup logger
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Analyze a Grasp Quality Convolutional Neural Network with TensorFlow')
    parser.add_argument('--output_dir', type=str, default='./', help='path to save the analysis')
    parser.add_argument('--config_filename', type=str, default=None, help='path to the configuration file to use')
    args = parser.parse_args()
    output_dir = args.output_dir
    config_filename = args.config_filename

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/analyze_gqcnn_performance.yaml')

    # turn relative paths absolute
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    # read config
    config = YamlConfig(config_filename)
        
    # run the analyzer
    logging.info('Starting analysis')
    analyzer = GQCNNAnalyzer(config)
    analyzer.analyze(output_dir)
