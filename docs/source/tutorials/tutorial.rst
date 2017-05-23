General Workflow
~~~~~~~~~~~~~~~~
The essence of the GQCNN module is to allow modular training of Grasp Quality
Neural Networks. The main idea is to be able to create a Grasp Quality Neural Network
and train it using a DeepOptimizer object. Once a GQCNN is trained it can be used to run grasp quality predictions.
Another key idea is the ability to benchmark the performance of GQCNN's using the GQCNNAnalyzer.  

Setup
~~~~~

Imports
+++++++
The following examples assume the GQCNN, DeepOptimizer, GQCNNAnalyzer, and YamlConfig objects have already been imported. An example import::

	from gqcnn import GQCNN, DeepOptimizer, GQCNNAnalyzer
	from core import YamlConfig

Configurations
++++++++++++++
For the following examples we will also assume we have the following configuration files::
	
	train_config = YamlConfig('path/to/training/configuration') # Sample config: 'cfg/tools/train_grasp_quality_cnn.yaml'
	gqcnn_config = train_config['gqcnn_config']
	analysis_config = YamlConfig('path/to/analysis/config') # Sample config: 'cfg/tools/analyze_gqcnn_performance.yaml'
	model_dir = '/path/to/model/dir'

All of the constructors in the GQCNN module expect configurations in the form of a dictionary so we have the freedom
to store our hyperparameters in any form that can be converted to a dictionary. The network configurations and architecture are a subset of the training configuration, hence in this example we are getting them from train_config. The
module is designed this way because architecture and training are closely linked and this way both are stored in one configuration file for simplicity.

Dataset
+++++++
A small sample dataset can be downloaded from `https://berkeley.app.box.com/s/p85ov4dx7vbq6y1l02gzrnsexg6yyayb/1/27311630602/` as `dexnet_2.0_adversarial.zip`. The overall download size is approximately 70MB. Once you have downloaded the dataset, unzipped it, and moved it to where you want, you `must` modify the `dataset_dir` parameter in the training configuration file(ex. train_grasp_quality_cnn.yaml)::

	dataset_dir: /your/path/to/dataset

Training a Network from Scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are two main steps to training a network from scratch:

1) Initialize a GQCNN and a DeepOptimizer to train it::

	gqcnn = GQCNN(gqcnn_config)
	deepOptimizer = DeepOptimizer(gqcnn, train_config)

2) Train the GQCNN::
	
	with gqcnn.get_tf_graph().as_default():
	     deepOptimizer.optimize()

Prediction
~~~~~~~~~~
Once we have trained a model predicting is simply a matter of instantiating a GQCNN with that model and running predictions::
	
	images = ['array of images']
	poses = ['corresponding poses']

	gqcnn = GQCNN.load(model_dir)
	gqcnn.predict(images, poses)
	gqcnn.close_session()

To predict multiple images we could load them from a file directory and call
the predict function in a loop.

Analysis
~~~~~~~~
Finally we can analyze models we have trained using the GQCNNAnalyzer::

	analyzer = GQCNNAnalyzer(analysis_config)
	analyzer.analyze()

The analysis_config contains a list of models to analyze at once along with many analysis parameters.

Fine-Tuning a Network
~~~~~~~~~~~~~~~~~~~~~
Fine tuning a network is similar to training one from scratch. The only difference is that we load a GQCNN from a model directory instead of creating one from scratch::

	gqcnn = GQCNN.load(model_dir)
	deepOptimizer = DeepOptimizer(gqcnn, train_config)
	with gqcnn.get_tf_graph().as_default():
	     deepOptimizer.optimize()

Visualizing Training with Tensorboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The DeepOptimizer is designed with support for Tensorboard to allow for visualization of various training 
parameters such as learning rate, validation error, minibatch loss, and minibatch error. These tensorboard summaries are 
saved in a folder labeled `tensorboard_summaries` in the model directory. For example, if the model directory where the model is being saved is `/home/user/Data/models/grasp_quality/model_qwueio`, the summaries will be stored in `/home/user/Data/models/grasp_quality/model_qwueio/tensorboard_summaries`. 

The DeepOptimizer will automatically start a local server to feed these summaries. Once you see this output message, `Launching Tensorboard, Please navigate to localhost:6006 in your favorite web browser to view summaries`, simply navigate to `localhost:6006` in your favorite web-browser to start visualizing.
