"""  
Script for running inference with the FC-GQ-CNN on the Intel NCS. 

Author
------
Vishal Satish
"""
import argparse
import os
import time

import numpy as np
from mvnc import mvncapi as mvnc
import IPython as ip
import tensorflow as tf

from autolab_core import Logger
from gqcnn import get_fc_gqcnn_model

DEPTH = 0.5
IM_HEIGHT = 180 # used for random input generation
IM_WIDTH = 320
OUTPUT_SHAPE = (1, 75, 145, 32)
#OUTPUT_SHAPE = (1, 32, 75, 145)
TOLERANCE = 0.000001
banner_half = ''.join(['#']*15)

np.random.seed(12345)

# setup logger
logger = Logger.get_logger('tools/run_inference_fcgqcnn_ncs.py')

def run_inference_ncs(im, depth, model_dir):
    def pairwise_softmax(tensor):
        """ Applies pair-wise softmax to dim -1 of tensor. """
        tensor_soft = np.copy(tensor)
        for i in xrange(0, tensor_soft.shape[-1], 2):
            tensor_soft[..., i:i+2] = np.exp(tensor_soft[..., i:i+2]) / np.sum(np.exp(tensor_soft[..., i:i+2]), axis=-1, keepdims=True) 
        return tensor_soft

    logger.info('{0} Running Inference on NCS {0}'.format(banner_half))

    logger.info('Loading image normalization metrics...')
    im_mean = np.load(os.path.join(model_dir, 'im_depth_sub_mean.npy')) 
    im_std = np.load(os.path.join(model_dir, 'im_depth_sub_std.npy'))

    logger.info('Subtracting depth...')
#    im -= depth

    logger.info('Pre-processing image...')
#    im = (im - im_mean) / im_std

#    im = im.reshape(1, 1, 180, 320)

    logger.info('Truncating image to 32-bit precision...')
#    im = im.astype(np.float32) #NOTE: This is b/c the default NCS FIFO queues are 32-bit
    im = im.astype(np.float16)    
#    im = np.transpose(im, (0, 3, 1, 2)) #TODO: @Vishal does it really have to be transposed?? 
 
    logger.info('Initializing NCS device...')
    device = mvnc.Device(mvnc.enumerate_devices()[0])
    device.open()

    logger.info('Loading graph onto device...')
    with open(os.path.join(model_dir, 'model.graph')) as fhandle:
        g_buff = fhandle.read()
    graph = mvnc.Graph('graph')
#    q_input_im, q_pred = graph.allocate_with_fifos(device, g_buff)
    graph.allocate(device, g_buff)
    fifoIn = mvnc.Fifo("fifoIn0", mvnc.FifoType.HOST_WO)
    fifoOut = mvnc.Fifo("fifoOut0", mvnc.FifoType.HOST_RO)
    fifoIn.set_option(mvnc.FifoOption.RW_DATA_TYPE, mvnc.FifoDataType.FP16)
    fifoOut.set_option(mvnc.FifoOption.RW_DATA_TYPE, mvnc.FifoDataType.FP16)
    descIn = graph.get_option(mvnc.GraphOption.RO_INPUT_TENSOR_DESCRIPTORS)
    descOut = graph.get_option(mvnc.GraphOption.RO_OUTPUT_TENSOR_DESCRIPTORS)
    ip.embed()
    fifoIn.allocate(device, descIn[0], 2)
    fifoOut.allocate(device, descOut[0], 2)
    q_input_im = fifoIn
    q_pred = fifoOut

    logger.info('Beginning inference...')
    inf_start_time = time.time()
    graph.queue_inference_with_fifo_elem(q_input_im, q_pred, im, None)
    preds, _ = q_pred.read_elem()
    logger.info('NCS inference took {} seconds.'.format(time.time() - inf_start_time))

    logger.info('Re-shaping predictions...')
#    preds = preds.reshape(OUTPUT_SHAPE) #TODO: @Vishal confirm that the default output (and also input) of inference are channel minor (NHWC)

    logger.info('Applying pair-wise softmax...')
#    preds = pairwise_softmax(preds)

    logger.info('Cleaning up...')
    q_input_im.destroy()
    q_pred.destroy()
    graph.destroy()
    device.close()
 
    return preds

def run_inference_tf(im, depth, model_dir):
    logger.info('{0} Running Inference with TF {0}'.format(banner_half))

#    im = im.astype(np.float32)
    im = im.astype(np.float16)

    fcgqcnn = get_fc_gqcnn_model().load(model_dir, {'im_height': im.shape[1], 'im_width': im.shape[2]})
    fcgqcnn.open_session()
    inf_start_time = time.time()
    preds = fcgqcnn.predict(im, np.asarray([depth]))
    logger.info('TF inference took {} seconds.'.format(time.time() - inf_start_time))
    fcgqcnn.close_session()
    
    return preds

def run_inference_tf_from_pb(im, depth, model_dir):
    im = im.astype(np.float16)
    from tensorflow.core.framework import graph_pb2
    session = tf.Session()
    with session.as_default():
        graph_def = graph_pb2.GraphDef()
        with open(os.path.join(model_dir, 'model.pb'), 'rb') as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
        model = tf.get_default_graph()    
        input_op = model.get_operation_by_name('input_im')
        output_op = model.get_operation_by_name('softmax/output')
        tf.global_variables_initializer()
        preds = session.run(output_op.outputs[0], feed_dict={input_op.outputs[0]: im})
        ip.embed()
    session.close()
    return preds

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description='Run inference with the FC-GQ-CNN on the Intel NCS')
    parser.add_argument('model_dir', type=str, default=None,
                        help='path to the GQ-CNN model containing a "model.graph" file representing a FC-GQ-CNN compiled with the Intel NCSDK')
    parser.add_argument('--input_im', type=str, default=None, 
                        help='path to the input image to query the network with; if not provided, a random image will be used')
    parser.add_argument('--tf', action='store_true',
                        help='query the TensorFlow implementation and cross-reference NCS output against it')

    args = parser.parse_args()
    model_dir = args.model_dir
    input_im_path = args.input_im
    run_tf = args.tf

    if input_im_path is not None:
        logger.info('Loading input image...')
        input_im = np.load(input_im_path)['arr_0']
    else:
        logger.info('Generating random input...')
        input_im = np.random.random_sample((1, IM_HEIGHT, IM_WIDTH, 1))
   
    logger.info('')
    preds_ncs = run_inference_ncs(input_im, DEPTH, model_dir)

    if run_tf:
        logger.info('')
        preds_tf = run_inference_tf_from_pb(input_im, DEPTH, model_dir)
    
        ip.embed()
        # validate NCS predictions against TF baseline
        assert preds_ncs.shape == preds_tf.shape, 'NCS and TF predictions have differing shapes of {} and {}, respectively'.format(preds_ncs.shape, preds_tf.shape)
        error = np.sum(np.abs(preds_ncs - preds_tf))
        assert error < TOLERANCE, 'Error between NCS predictions and TF predictions was {}, which is above allowed tolerance of {}!'.format(error, TOLERANCE)
