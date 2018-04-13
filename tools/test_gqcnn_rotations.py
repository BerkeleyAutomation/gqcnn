# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Quick file to test GQ-CNN rotations.
Author: Jeff Mahler
"""
import argparse
import cv2
import IPython
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import os

import tensorflow as tf

from autolab_core import Point, YamlConfig
import autolab_core.utils as utils
from gqcnn import GQCNN, SGDOptimizer, GQCNNAnalyzer, Grasp2D
from perception import DepthImage
from dexnet.visualization import DexNetVisualizer2D as vis2d

if __name__ == '__main__':
    # setup logger
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Train a Grasp Quality Convolutional Neural Network with TensorFlow')
    parser.add_argument('model_dir', type=str, default=None, help='path to the GQ-CNN weights')
    parser.add_argument('dataset_dir', type=str, default=None, help='dataset to test on')
    parser.add_argument('num_rots', type=int, default=None, help='number of rotations to evaluate')
    args = parser.parse_args()
    model_dir = args.model_dir
    dataset_dir = args.dataset_dir
    num_rots = args.num_rots

    plot = False
    
    # load model
    gqcnn = GQCNN.load(model_dir)

    # open dataset
    file_num = 6300
    ims_filename = os.path.join(dataset_dir, 'depth_ims_tf_table_%05d.npz' %(file_num))
    poses_filename = os.path.join(dataset_dir, 'hand_poses_%05d.npz' %(file_num))
    metrics_filename = os.path.join(dataset_dir, 'robust_ferrari_canny_%05d.npz' %(file_num))
    ims = np.load(ims_filename)['arr_0']
    poses = np.load(poses_filename)['arr_0']
    metrics = np.load(metrics_filename)['arr_0']
    labels = 1 * (metrics > 0.002)

    batch_size = 16
    ims = ims[:batch_size,...]
    poses = poses[:batch_size,2:3]
    labels = labels[:batch_size]

    predictions = {}
    
    # open session
    with gqcnn.graph.as_default():
        sess = gqcnn.open_session()
        orig_features = gqcnn.featurize(ims, poses, feature_layer='conv1_1')

        theta = 0
        orig_convW = {}
        for k in range(num_rots):
            theta = float(k) * 180.0 / num_rots
            logging.info('Evaluating network at rotation %.3f' %(theta))

            for layer in gqcnn.feature_tensors.keys():
                if layer.find('conv') == -1:
                    continue
                    
                logging.info('Rotating filters for %s' %(layer))

                convW_var = gqcnn.weights.__dict__['%sW' %(layer)]
                if layer not in orig_convW.keys():
                    orig_convW[layer] = sess.run(convW_var)
                convW = orig_convW[layer]     
                height = convW.shape[0]
                width = convW.shape[1]
                num_in_channels = convW.shape[2]
                num_out_channels = convW.shape[3]
                f_center = [float(height-1) / 2, float(width-1) / 2]
                rot_map = cv2.getRotationMatrix2D((f_center[1], f_center[0]), -theta, 1)
                new_convW = np.zeros(convW.shape)

                for i in range(num_in_channels):
                    for j in range(num_out_channels):
                        orig_filt = convW[:,:,i,j]
                        rot_filt = cv2.warpAffine(orig_filt, rot_map, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                        new_convW[:,:,i,j] = rot_filt

                if plot and layer == 'conv1_1':
                    d = utils.sqrt_ceil(num_out_channels)
                    plt.figure()
                    for i in range(num_out_channels):
                        plt.subplot(d,d,i+1)
                        plt.imshow(convW[:,:,0,i], cmap=plt.cm.gray_r)
                        plt.axis('off')
                    plt.figure()
                    for i in range(num_out_channels):
                        plt.subplot(d,d,i+1)
                        plt.imshow(new_convW[:,:,0,i], cmap=plt.cm.gray_r)
                        plt.axis('off')
                    plt.show()
                    
                sess.run(convW_var.assign(new_convW))
                #gqcnn._output_tensor = gqcnn._build_network(gqcnn._input_im_node,
                #                                            gqcnn._input_pose_node)
                #sess.run(tf.global_variables_initializer())
                #new_features = gqcnn.featurize(ims, poses, feature_layer='conv1_1')

            # predict the images
            p_success = gqcnn.predict(ims, poses)[:,1]
            predictions[theta] = p_success
            
        # plot        
        vis2d.figure()
        k = 1
        thetas = predictions.keys()
        thetas.sort()
        for theta in thetas:
            p_success = predictions[theta]
            for j in range(batch_size):
                im = DepthImage(ims[j,:,:,0])
                grasp = Grasp2D(im.center,
                                np.deg2rad(theta),
                                depth=poses[j],
                                width=13.5)
                vis2d.subplot(num_rots, batch_size, k)
                vis2d.imshow(im)
                vis2d.grasp(grasp, scale=0.25)
                vis2d.title('P(S) = %.4f' %(p_success[j]), fontsize=5)
                k += 1
        vis2d.show()

        gqcnn.close_session()
