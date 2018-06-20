import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sm
import cv2 as cv

from gqcnn.model import get_gqcnn_model
from gqcnn import Visualizer as vis, Grasp2D

#MODEL_DIR = '/home/vsatish/Data/dexnet/data/models/test_dump/model_mgedsvljcu'
#FULLY_CONV_CONFIG = {'im_width': 500, 'im_height': 500}
#RECEP_H = 96
#RECEP_W = 96

#Coke
#DEPTH = np.asarray([[0.723]])
#DEPTH_IM = '/nfs/diskstation/vsatish/dex-net/data/datasets/kit_sub_dexnet_fc_06_13_18/images/tensors/depth_ims_00005.npz'
#ANG_BIN = 5

#Unknown
#DEPTH = np.asarray([[0.75]])
#DEPTH_IM = '/nfs/diskstation/vsatish/dex-net/data/datasets/kit_sub_dexnet_fc_06_13_18/images/tensors/depth_ims_00001.npz'
#ANG_BIN = 14

#Heart
#DEPTH = np.asarray([[0.713]])
#DEPTH_IM = '/nfs/diskstation/vsatish/dex-net/data/datasets/kit_sub_dexnet_fc_06_13_18/images/tensors/depth_ims_00002.npz'
#ANG_BIN = 6
#ANG_BIN = 0

#Shampoo
#DEPTH = np.asarray([[0.755]])
#DEPTH_IM = '/nfs/diskstation/vsatish/dex-net/data/datasets/kit_sub_dexnet_fc_06_13_18/images/tensors/depth_ims_00003.npz'
#ANG_BIN = 12

#Sardines
#DEPTH = np.asarray([[0.748]])
#DEPTH_IM = '/nfs/diskstation/vsatish/dex-net/data/datasets/kit_sub_dexnet_fc_06_13_18/images/tensors/depth_ims_00004.npz'
#ANG_BIN = 8

#Cat
#DEPTH = np.asarray([[0.7]])
#DEPTH_IM = '/nfs/diskstation/vsatish/dex-net/data/datasets/kit_sub_dexnet_fc_06_13_18/images/tensors/depth_ims_00007.npz'
#ANG_BIN = 7

#Sunflower Oil
#DEPTH = np.asarray([[0.755]])
#DEPTH_IM = '/nfs/diskstation/vsatish/dex-net/data/datasets/kit_sub_dexnet_fc_06_13_18/images/tensors/depth_ims_00009.npz'
#ANG_BIN = 14

#Unknown
#DEPTH = np.asarray([[0.78]])
#DEPTH_IM = '/nfs/diskstation/vsatish/dex-net/data/datasets/kit_sub_dexnet_fc_06_13_18/images/tensors/depth_ims_00000.npz'
#ANG_BIN = 14

#Unknown-need to use [50:51]
#DEPTH = np.asarray([[0.719]])
#DEPTH_IM = '/nfs/diskstation/vsatish/dex-net/data/datasets/kit_sub_dexnet_fc_06_13_18/images/tensors/depth_ims_00001.npz'
#ANG_BIN = 8

#Fizzytablets-need to use [50:51]
#DEPTH = np.asarray([[0.76]])
#DEPTH_IM = '/nfs/diskstation/vsatish/dex-net/data/datasets/kit_sub_dexnet_fc_06_13_18/images/tensors/depth_ims_00003.npz'
#ANG_BIN = 12

#im = np.load(DEPTH_IM)['arr_0'][0:1]
#plt.figure()
#plt.imshow(im[0, :, :, 0], cmap='gray')
#plt.show()

#fully_conv_gqcnn = get_gqcnn_model().load(MODEL_DIR, fully_conv_config=FULLY_CONV_CONFIG)
#fully_conv_gqcnn.open_session()
#pred = fully_conv_gqcnn.predict(im, DEPTH)
#fully_conv_gqcnn.close_session()
#import IPython
#IPython.embed()

#im_h, im_w = FULLY_CONV_CONFIG['im_height'], FULLY_CONV_CONFIG['im_width']
#cropped_orig = im[0, ..., 0][RECEP_H // 2:im_h - (RECEP_H // 2), RECEP_W // 2:im_w - (RECEP_W // 2)]
#downsamp_orig = cv.resize(cropped_orig, (pred.shape[1], pred.shape[2]), interpolation=cv.INTER_NEAREST)
#plt.figure()
#plt.subplot(121)
#plt.imshow(downsamp_orig, cmap='gray')
#plt.subplot(122)
#plt.imshow(downsamp_orig, cmap='gray')
#plt.imshow(pred[0, :, :, ANG_BIN * 2 + 1], alpha=0.9, cmap='RdYlGn', interpolation='nearest')
#vis.grasp(Grasp2D)
#plt.show()


