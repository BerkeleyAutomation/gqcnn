import os
import numpy as np
import IPython as ip
from gqcnn.model import get_gqcnn_model
import matplotlib.pyplot as plt
import logging
import cv2
 
MODEL_DIR = '/home/vsatish/Data/dexnet/data/models/test_dump/model_xxyappgzwt'
# MODEL_DIR = '/home/vsatish/Data/dexnet/data/models/test_dump/model_ebhsmdqmjd'
DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/dexnet_2_fcn'
IM_FILENAME = 'depth_ims_tf_table_00000.npz'
POSE_FILENAME = 'hand_poses_00000.npz'
FULLY_CONV_CONFIG = {'im_width': 46, 'im_height': 46}
IM_INDEX = 0
FEAT = 'conv2_2'

# restart and setup logger
logging.shutdown()
reload(logging)
logging.getLogger().setLevel(logging.INFO)

# load test image and pose
im = np.load(os.path.join(DATASET_DIR, IM_FILENAME))['arr_0'][IM_INDEX:IM_INDEX+1]
pose = np.load(os.path.join(DATASET_DIR, POSE_FILENAME))['arr_0'][IM_INDEX:IM_INDEX+1, 2:3]

# generate rotated version of image
rot_im = np.zeros_like(im)
rot_mat = cv2.getRotationMatrix2D(((im.shape[2] - 1) / 2, (im.shape[1] - 1) / 2), 45, 1)
rot_im[0, :, :, 0] = cv2.warpAffine(im[0, :, :, 0], rot_mat, (im.shape[2], im.shape[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
#rot_im[0, :, :, 0] = np.copy(np.rot90(im[0, :, :, 0], k=1))

# create normal fc-gqcnn
normal_fully_conv_gqcnn = get_gqcnn_model().load(MODEL_DIR, fully_conv_config=FULLY_CONV_CONFIG)

# create rot fc-gqcnn
rot_fully_conv_gqcnn = get_gqcnn_model().load(MODEL_DIR, fully_conv_config=FULLY_CONV_CONFIG, conv_filt_rot=45)

# run predictions using normal fc-gqcnn
normal_fully_conv_gqcnn.open_session()
normal_pred = normal_fully_conv_gqcnn.predict(im, pose)
normal_feat = normal_fully_conv_gqcnn.featurize(im, pose, feature_layer=FEAT)
#rot_test = normal_fully_conv_gqcinn.featurize(im, pose, feature_layer='conv2_2_rot_test')
with normal_fully_conv_gqcnn._graph.as_default():
    tmp = normal_fully_conv_gqcnn._sess.run(normal_fully_conv_gqcnn._weights.weights['conv1_1_weights'])[:, :, 0, 7]
normal_fully_conv_gqcnn.close_session()

# run predictions using rot fc-gqcnn
rot_fully_conv_gqcnn.open_session()
rot_pred = rot_fully_conv_gqcnn.predict(rot_im, pose)
rot_feat = rot_fully_conv_gqcnn.featurize(rot_im, pose, feature_layer=FEAT)
with rot_fully_conv_gqcnn._graph.as_default():
    tmp1 = rot_fully_conv_gqcnn._sess.run(rot_fully_conv_gqcnn._weights.weights['conv1_1_weights'])[:, :, 0, 7]
rot_fully_conv_gqcnn.close_session()

#plt.figure()
#plt.subplot(121)
#plt.imshow(tmp, cmap='gray')
#plt.subplot(122)
#plt.imshow(tmp1, cmap='gray')
#plt.show()

plt.figure()
plt.subplot(121)
plt.imshow(normal_feat[0, :, :, 0])
plt.subplot(122)
plt.imshow(rot_feat[0, :, :, 0])
#plt.subplot(133)
#plt.imshow(rot_test[0, :, :, 0])
plt.show()

# now rotate features back
rot_rot_feat = np.zeros_like(rot_feat)
for i in range(rot_feat.shape[3]):
    rot_rot_feat[0, :, :, i] = np.copy(np.rot90(rot_feat[0, :, :, i], k=-1))

# compare
print(np.max(np.abs(normal_feat - rot_rot_feat)))
print(normal_pred, rot_pred)
