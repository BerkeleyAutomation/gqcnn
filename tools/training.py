"""
GQCNN training script using DeepOptimizer
Author: Vishal Satish
"""
import time
import logging

from gqcnn import GQCNNAnalyzer
from gqcnn.models import get_gqcnn_model
from gqcnn.training import get_gqcnn_trainer
from autolab_core import YamlConfig
	
# setup logger
logging.getLogger().setLevel(logging.INFO)

train_config = YamlConfig('cfg/tools/training.yaml')
gqcnn_config = train_config['gqcnn_config']
backend = 'tf'

def get_elapsed_time(time_in_seconds):
	""" Helper function to get elapsed time """
	if time_in_seconds < 60:
		return '%.1f seconds' % (time_in_seconds)
	elif time_in_seconds < 3600:
		return '%.1f minutes' % (time_in_seconds / 60)
	else:
		return '%.1f hours' % (time_in_seconds / 3600)

###Possible Use-Cases###

# Training from Scratch
start_time = time.time()
gqcnn = get_gqcnn_model(backend)(gqcnn_config)
gqcnn_trainer = get_gqcnn_trainer(backend)(gqcnn, train_config)
gqcnn_trainer.train()
logging.info('Total Training Time:' + str(get_elapsed_time(time.time() - start_time))) 


# Prediction

# start_time = time.time()
# model_dir = '/home/user/Data/models/grasp_quality/model_ewlohgukns'
# gqcnn = get_gqcnn_model(backend).load(model_dir)
# output = gqcnn.predict(images, poses)
# pred_p_success = output[:,1]
# gqcnn.close_session()
# logging.info('Total Prediction Time:' + str(get_elapsed_time(time.time() - start_time)))


# Analysis

# start_time = time.time()
# analysis_config = YamlConfig('cfg/tools/analyze_gqcnn_performance.yaml')
# analyzer = GQCNNAnalyzer(analysis_config)
# analyzer.analyze()
# logging.info('Total Analysis Time:' + str(get_elapsed_time(time.time() - start_time)))


# Fine-Tuning

# start_time = time.time()
# model_dir = '/home/user/Data/models/grasp_quality/model_ewlohgukns'
# gqcnn = GQCNN.load(model_dir)
# deepOptimizer = DeepOptimizer(gqcnn, train_config)
# with gqcnn._graph.as_default():
    # deepOptimizer.optimize()
# logging.info('Total Fine Tuning Time:' + str(get_elapsed_time(time.time() - start_time)))

