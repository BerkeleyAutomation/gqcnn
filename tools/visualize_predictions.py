"""
Script for visualizing GQCNN predictions.
Author: Vishal Satish
"""

import logging

from core import YamlConfig
from gqcnn import GQCNNPredictionVisualizer
	
# setup logger
logging.getLogger().setLevel(logging.INFO)

# load a valid config
visualization_config = YamlConfig('cfg/tools/gqcnn_prediction_visualizer.yaml')

logging.info('Beginning Visualization')
visualizer = GQCNNPredictionVisualizer(visualization_config)
visualizer.visualize()