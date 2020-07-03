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

Analyzes a GQ-CNN model.

Author
------
Vishal Satish & Jeff Mahler
"""
import argparse
import os

from autolab_core import YamlConfig, Logger
from gqcnn import GQCNNAnalyzer

# Setup logger.
logger = Logger.get_logger("tools/analyze_gqcnn_performance.py")

if __name__ == "__main__":
    # Parse args.
    parser = argparse.ArgumentParser(
        description=("Analyze a Grasp Quality Convolutional Neural Network"
                     " with TensorFlow"))
    parser.add_argument("model_name",
                        type=str,
                        default=None,
                        help="name of model to analyze")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="path to save the analysis")
    parser.add_argument(
        "--dataset_config_filename",
        type=str,
        default=None,
        help="path to a configuration file for testing on a custom dataset")
    parser.add_argument("--config_filename",
                        type=str,
                        default=None,
                        help="path to the configuration file to use")
    parser.add_argument("--model_dir",
                        type=str,
                        default=None,
                        help="path to the model")
    args = parser.parse_args()
    model_name = args.model_name
    output_dir = args.output_dir
    dataset_config_filename = args.dataset_config_filename
    config_filename = args.config_filename
    model_dir = args.model_dir

    # Create model dir.
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "../models")
    model_dir = os.path.join(model_dir, model_name)

    # If `model_dir` contains many models, analyze all of them.
    model_dir = [model_dir]
    if "config.json" not in os.listdir(model_dir[0]):
        logger.warning(
            "Found multiple models in model_dir, analyzing all of them...")
        models = os.listdir(model_dir[0])
        model_dir = [os.path.join(model_dir[0], model) for model in models]

    # Set defaults.
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "../analysis")
    if config_filename is None:
        config_filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..",
            "cfg/tools/analyze_gqcnn_performance.yaml")

    # Turn relative paths absolute.
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)
    if dataset_config_filename is not None and not os.path.isabs(
            dataset_config_filename):
        dataset_config_filename = os.path.join(os.getcwd(),
                                               dataset_config_filename)

    # Make the output dir.
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Read config.
    config = YamlConfig(config_filename)

    dataset_config = None
    if dataset_config_filename is not None:
        dataset_config = YamlConfig(dataset_config_filename)

    # Run the analyzer.
    analyzer = GQCNNAnalyzer(config, plot_backend="pdf")
    for model in model_dir:
        analyzer.analyze(model, output_dir, dataset_config)
