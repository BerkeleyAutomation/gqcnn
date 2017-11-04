"""
GQCNN training script using DeepOptimizer
Author: Vishal Satish
"""
from gqcnn import GQCNN, SGDOptimizer, GQCNNAnalyzer
from autolab_core import YamlConfig
import time
import logging
	
#setup logger
logging.getLogger().setLevel(logging.INFO)

train_config = YamlConfig('cfg/tools/training.yaml')
gqcnn_config = train_config['gqcnn_config']

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
#start_time = time.time()
#gqcnn = GQCNN(gqcnn_config)
#sgdOptimizer = SGDOptimizer(gqcnn, train_config)
#with gqcnn.get_tf_graph().as_default():
#    sgdOptimizer.optimize()
#logging.info('Total Training Time:' + str(get_elapsed_time(time.time() - start_time))) 


# Prediction

# start_time = time.time()
# model_dir = '/home/user/Data/models/grasp_quality/model_ewlohgukns'
# gqcnn = GQCNN.load(model_dir)
# output = gqcnn.predict(images, poses)
# pred_p_success = output[:,1]
# gqcnn.close_session()
# logging.info('Total Prediction Time:' + str(get_elapsed_time(time.time() - start_time)))


# Analysis

start_time = time.time()
analysis_config = YamlConfig('cfg/tools/analyze_gqcnn_performance.yaml')
analyzer = GQCNNAnalyzer(analysis_config)
analyzer.analyze()
logging.info('Total Analysis Time:' + str(get_elapsed_time(time.time() - start_time)))


# Fine-Tuning

# start_time = time.time()
# model_dir = '/home/user/Data/models/grasp_quality/model_ewlohgukns'
# gqcnn = GQCNN.load(model_dir)
# deepOptimizer = DeepOptimizer(gqcnn, train_config)
# with gqcnn._graph.as_default():
    # deepOptimizer.optimize()
# logging.info('Total Fine Tuning Time:' + str(get_elapsed_time(time.time() - start_time)))

