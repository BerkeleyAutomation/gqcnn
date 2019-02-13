Analysis
~~~~~~~~
It is helpful to check the training and validation loss and classification errors to ensure that the network has trained successfully. To analyze the performance of a trained GQ-CNN, run: ::

    $ python tools/analyze_gqcnn_performance.py <model_name>

The args are:

#. **model_name**: Name of a trained model.

The script will store a detailed analysis in `analysis/<model_name>/`.

To analyze the networks we just trained, run: ::

    $ python tools/analyze_gqcnn_performance.py gqcnn_example_pj
    $ python tools/analyze_gqcnn_performance.py gqcnn_example_suction

Below is the expected output for the **parallel jaw** network. Please keep in mind that the exact performance values may change due to randomization in the training dataset and random weight initialization: ::

    $ GQCNNAnalyzer INFO     TRAIN
    $ GQCNNAnalyzer INFO     Original error: 36.812
    $ GQCNNAnalyzer INFO     Final error: 6.061
    $ GQCNNAnalyzer INFO     Orig loss: 0.763
    $ GQCNNAnalyzer INFO     Final loss: 0.248
    $ GQCNNAnalyzer INFO     VAL
    $ GQCNNAnalyzer INFO     Original error: 32.212
    $ GQCNNAnalyzer INFO     Final error: 7.509
    $ GQCNNAnalyzer INFO     Normalized error: 0.233

A set of plots will be saved to `analysis/gqcnn_example_pj/`. The plots `training_error_rates.png` and `precision_recall.png` should look like the following:

.. image:: ../images/plots/pj_error_rate.png
   :width: 49 %

.. image:: ../images/plots/pj_roc.png
   :width: 49 %

Here is the expected output for the **suction** network: ::

    $ GQCNNAnalyzer INFO     TRAIN
    $ GQCNNAnalyzer INFO     Original error: 17.844
    $ GQCNNAnalyzer INFO     Final error: 6.417
    $ GQCNNAnalyzer INFO     Orig loss: 0.476
    $ GQCNNAnalyzer INFO     Final loss: 0.189
    $ GQCNNAnalyzer INFO     VAL
    $ GQCNNAnalyzer INFO     Original error: 18.036
    $ GQCNNAnalyzer INFO     Final error: 6.907
    $ GQCNNAnalyzer INFO     Normalized error: 0.383

A set of plots will be saved to `analysis/gqcnn_example_suction/`. The plots `training_error_rates.png` and `precision_recall.png` should look like the following:

.. image:: ../images/plots/suction_error_rate.png
   :width: 49 %

.. image:: ../images/plots/suction_roc.png
   :width: 49 %

