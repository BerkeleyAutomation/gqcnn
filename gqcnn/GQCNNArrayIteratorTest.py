from gqcnn_predict_iterator import GQCNNPredictIterator
from neon.backends import gen_backend
import numpy as np
import IPython

be = gen_backend(backend='gpu', batch_size=64)

images = np.load('/mnt/hdd/dex-net/data/gqcnn/grasp_quality/mini_dexnet_all_trans_01_20_17/depth_ims_tf_00070.npz')['arr_0']
poses = np.load('/mnt/hdd/dex-net/data/gqcnn/grasp_quality/mini_dexnet_all_trans_01_20_17/hand_poses_00156.npz')['arr_0']
images = images.reshape((1000, 1024))

images = images[:5]
poses = poses[:5]

my_iter = GQCNNPredictIterator(images, poses, lshape=[(32, 32, 1), (4,)])
outputs = [output for output in my_iter]
IPython.embed()
