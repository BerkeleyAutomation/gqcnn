import time

from autolab_core import YamlConfig
from gqcnn import GQCNNAnalyzer

start_time = time.time()
analysis_config = YamlConfig('cfg/tools/analyze_gqcnn_performance.yaml')
analyzer = GQCNNAnalyzer(analysis_config)
analyzer.analyze()
logging.info('Total Analysis Time:' + str(get_elapsed_time(time.time() - start_time)))