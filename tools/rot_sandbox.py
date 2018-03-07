import os
import numpy as np
import IPython as ip
from gqcnn.model import get_gqcnn_model
import matplotlib.pyplot as plt
import logging
import cv2

MODEL_DIR = '/home/vsatish/Data/dexnet/data/models/test_dump/model_ebhsmdqmjd'
DATASET_DIR = '/nfs/diskstation/vsatish/dex-net/data/datasets/dexnet_2_fcn'
IM_FILENAME = 'depth_ims_tf_table_00000.npz'
POSE_FILENAME = 'hand_poses_00000.npz'
FULLY_CONV_CONFIG = {'im_width': 46, 'im_height': 46}
IM_INDEX = 0
FEAT = 'conv2_2'
ROT_BUCKETS = 16

# restart and setup logger
logging.shutdown()
reload(logging)
logging.getLogger().setLevel(logging.INFO)

# load test image and pose
im = np.load(os.path.join(DATASET_DIR, IM_FILENAME))['arr_0'][IM_INDEX, :, :, 0]
pose = np.load(os.path.join(DATASET_DIR, POSE_FILENAME))['arr_0'][IM_INDEX:IM_INDEX+1, 2:3]

# allocate image tensors
images = np.zeros((ROT_BUCKETS, 46, 46, 1))
rot_images = np.zeros((ROT_BUCKETS, 46, 46, 1))

# generate rotated versions of image
for i in range(ROT_BUCKETS):
    rot_mat = cv2.getRotationMatrix2D(((im.shape[1] - 1) / 2, (im.shape[0] - 1) / 2), 180.0 / ROT_BUCKETS * i, 1)
    rot_image = cv2.warpAffine(im, rot_mat, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rot_images[i, :, :, 0] = rot_image
    images[i, :, :, 0] = im

#plt.figure()
#plt.subplot(121)
#plt.imshow(images[0, :, :, 0], cmap='gray')
#plt.subplot(122)
#plt.imshow(rot_images[8, :, :, 0], cmap='gray')
#plt.show()

# create normal fc-gqcnn
normal_fully_conv_gqcnn = get_gqcnn_model().load(MODEL_DIR, fully_conv_config=FULLY_CONV_CONFIG)

# create rotated fc-gqcnns
rot_fcgqcnns = []
for i in range(ROT_BUCKETS):
    rot_fcgqcnns.append(get_gqcnn_model().load(MODEL_DIR, fully_conv_config=FULLY_CONV_CONFIG, conv_filt_rot=180.0 / ROT_BUCKETS * i))

# run predictions using normal fc-gqcnn
normal_fully_conv_gqcnn.open_session()
normal_pred = normal_fully_conv_gqcnn.predict(images[0:1], pose)
normal_feat = normal_fully_conv_gqcnn.featurize(images[0:1], pose, feature_layer=FEAT)
normal_fully_conv_gqcnn.close_session()

# run predictions using rot fc-gqcnn
rot_pred = None
rot_feat = None
for i in range(ROT_BUCKETS):
    rot_fully_conv_gqcnn = rot_fcgqcnns[i]
    rot_fully_conv_gqcnn.open_session()
    pred = rot_fully_conv_gqcnn.predict(rot_images[i:i+1], pose)
    feat = rot_fully_conv_gqcnn.featurize(rot_images[i:i+1], pose, feature_layer=FEAT)
    rot_fully_conv_gqcnn.close_session()
    if rot_pred is None:
        rot_pred = np.zeros((ROT_BUCKETS,) + pred.shape[1:])
    if rot_feat is None:
        rot_feat = np.zeros((ROT_BUCKETS,) + feat.shape[1:])
    rot_pred[i] = pred[0]
    rot_feat[i] = feat[0]

# now rotate features back
rot_rot_feat = np.zeros_like(rot_feat)
for i in range(ROT_BUCKETS):
    rot_mat = cv2.getRotationMatrix2D(((rot_feat.shape[2] - 1) / 2, (rot_feat.shape[1] - 1) / 2), -1 * 180.0 / ROT_BUCKETS * i, 1)
    for j in range(rot_feat.shape[3]):
        rot_rot_feat[i, :, :, j] = cv2.warpAffine(rot_feat[i, :, :, j], rot_mat, (rot_feat.shape[2], rot_feat.shape[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# compare
for i in range(ROT_BUCKETS):
    ang = 180.0 / ROT_BUCKETS * i
    print("Angle: {}, NORMAL Pred: {}, ROT PRED: {}, DIVERGENCE: {}".format(ang, normal_pred, rot_pred[i], np.max(np.abs(normal_feat[0] - rot_rot_feat[i]))))
