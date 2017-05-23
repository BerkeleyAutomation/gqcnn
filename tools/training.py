"""
GQCNN training script using DeepOptimizer
Author: Vishal Satish
"""
from gqcnn import GQCNN, DeepOptimizer, GQCNNAnalyzer
from core import YamlConfig
import time
import logging
	
#setup logger
logging.getLogger().setLevel(logging.INFO)

# train_config = YamlConfig('cfg/tools/train_grasp_quality_cnn_dexnet_large.yaml')
train_config = YamlConfig('cfg/tools/train_micro_dex-net.yaml')
gqcnn_config = train_config['gqcnn_config']
analysis_config = YamlConfig('cfg/tools/analyze_gqcnn_performance.yaml')
model_dir = '/home/autolab/Public/data/dex-net/data/models/grasp_quality/gqcnn_vgg_mini_dexnet_robust_eps_replication_01_23_17'

def get_elapsed_time(time_in_seconds):
	""" Helper function to get elapsed time """
	if time_in_seconds < 60:
		return '%.1f seconds' % (time_in_seconds)
	elif time_in_seconds < 360:
		return '%.1f minutes' % (time_in_seconds / 60)
	else:
		return '%.1f hours' % (time_in_seconds / 360)

###Possible Use-Cases###

# Use Case 1-Prediction
# start_time = time.time()
# gqcnn = GQCNN.load(model_dir)
# gqcnn.predict(images, poses)
# gqcnn.close_session()
# logging.info('Total Prediction Time:' + str(get_elapsed_time(time.time() - start_time)))

# Use Case 2-Training from Scratch
start_time = time.time()
gqcnn = GQCNN(gqcnn_config)
deepOptimizer = DeepOptimizer(gqcnn, train_config)
with gqcnn.get_tf_graph().as_default():
    deepOptimizer.optimize()
logging.info('Total Training Time:' + str(get_elapsed_time(time.time() - start_time))) 

# Use Case 3-Fine Tuning
# start_time = time.time()
# gqcnn = GQCNN.load(model_dir)
# deepOptimizer = DeepOptimizer(gqcnn, train_config)
# with gqcnn._graph.as_default():
    # deepOptimizer.optimize()
# print 'Total Fine Tuning Time:', get_elapsed_time(time.time() - start_time)

# Analysis
start_time = time.time()
analyzer = GQCNNAnalyzer(analysis_config)
analyzer.analyze()
logging.info('Total Analysis Time:' + str(get_elapsed_time(time.time() - start_time)))
