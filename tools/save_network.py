import numpy as np
import os
import sys
import tensorflow as tf

from autolab_core import YamlConfig
from gqcnn import get_fc_gqcnn_model

if __name__ == '__main__':
    model_dir = sys.argv[1]
    fully_conv_config = YamlConfig(sys.argv[2])['policy']['metric']['fully_conv_gqcnn_config']
    output_dir = sys.argv[3]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    fcgqcnn = get_fc_gqcnn_model(backend='tf').load(model_dir, fully_conv_config)

    with fcgqcnn.tf_graph.as_default():
        saver = tf.train.Saver(tf.global_variables())
        sess = fcgqcnn.open_session()
        sess.run(tf.local_variables_initializer())

        saver.save(sess, os.path.join(output_dir, 'model.ckpt'))

        writer = tf.summary.FileWriter('output', sess.graph)
        image_arr = np.random.rand(10,480,640,1)
        pose_arr = np.random.rand(10)
        
        fcgqcnn.predict(image_arr, pose_arr)
        writer.close()

        fcgqcnn.close_session()
