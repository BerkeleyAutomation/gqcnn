"""
Script for visualizing GQCNN predictions.
Author: Vishal Satish
"""

import logging

from core import YamlConfig
from gqcnn import GQCNNPredictionVisualizer
	
#setup logger
logging.getLogger().setLevel(logging.INFO)

visualization_config = 'cfg/tools/gqcnn_prediction_visualizer.yaml'

logging.info('Beginning Visualization')
visualizer = GQCNNPredictionVisualizer(YamlConfig(visualization_config))
visualizer.visualize()