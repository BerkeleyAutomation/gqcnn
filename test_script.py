import IPython

from gqcnn import GQCNN
from autolab_core import YamlConfig

temp = GQCNN(YamlConfig('cfg/tools/training.yaml')['gqcnn_config'])
temp.initialize_network()
IPython.embed()