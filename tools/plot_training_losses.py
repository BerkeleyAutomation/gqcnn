"""
Script to plot the various errors saved during training.

Author
------
Jeff Mahler

Required Parameters
------------------------
model_dir : str
    Command line argument, the path to the model whose errors are to plotted. All plots and other metrics will
    be saved to this directory. 
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import h5py

DATABASE_FNAME = 'data.h5'

if __name__ == '__main__':
    result_dir = sys.argv[1]
    with h5py.File(os.path.join(result_dir, DATABASE_FNAME), 'r') as f:
        val_errors = f['metrics']['Error Rate'][:] * 100
    num_epochs = val_errors.shape[0]

    val_iters = np.arange(num_epochs)

    print ('Final val error: {}'.format(val_errors[-1]))

    plt.figure(figsize=(8, 6))
    plt.margins(x=0.05)
    plt.plot(val_iters, val_errors, linewidth=4, color='g')
    plt.ylim(0, 40)
    plt.legend(('Validation',), fontsize=15, loc='upper right')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Error Rate', fontsize=15)
    plt.savefig(os.path.join(result_dir, 'validation_error.png')) 
