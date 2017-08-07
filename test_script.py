import IPython
import numpy as np

from gqcnn import GQCNN, GQCNNDataset
from autolab_core import YamlConfig

from neon.backends import gen_backend
# gqcnn = GQCNN(YamlConfig('cfg/tools/training.yaml')['gqcnn_config'])
# gqcnn.initialize_network()
# IPython.embed()

# images = np.load('/mnt/hdd/dex-net/data/gqcnn/grasp_quality/mini_dexnet_all_trans_01_20_17/depth_ims_tf_00070.npz')['arr_0']
# poses = np.load('/mnt/hdd/dex-net/data/gqcnn/grasp_quality/mini_dexnet_all_trans_01_20_17/hand_poses_00156.npz')['arr_0']
# outputs = gqcnn.predict(images, poses)
# IPython.embed()

be = gen_backend(backend='gpu', batch_size=64)
gqcnn_dataset = GQCNNDataset(YamlConfig('cfg/tools/training.yaml'))
data_iterators = gqcnn_dataset.gen_iterators()
train_iterator = data_iterators['train']
val_iterator = data_iterators['test']

train_iterator.__iter__().next()
IPython.embed()
