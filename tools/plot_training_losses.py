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
import IPython
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

PCT_POS_VAL_FILENAME = 'pct_pos_val.npy'
TRAIN_LOSS_FILENAME = 'train_losses.npy'
TRAIN_ERRORS_FILENAME = 'train_errors.npy'
VAL_ERRORS_FILENAME = 'val_errors.npy'
TRAIN_ITERS_FILENAME = 'train_eval_iters.npy'
VAL_ITERS_FILENAME = 'val_eval_iters.npy'
WINDOW = 100

if __name__ == '__main__':
    result_dir = sys.argv[1]
    train_errors_filename = os.path.join(result_dir, TRAIN_ERRORS_FILENAME)
    val_errors_filename = os.path.join(result_dir, VAL_ERRORS_FILENAME)
    train_iters_filename = os.path.join(result_dir, TRAIN_ITERS_FILENAME)
    val_iters_filename = os.path.join(result_dir, VAL_ITERS_FILENAME)
    pct_pos_val_filename = os.path.join(result_dir, PCT_POS_VAL_FILENAME)
    train_losses_filename = os.path.join(result_dir, TRAIN_LOSS_FILENAME)

    raw_train_errors = np.load(train_errors_filename)
    val_errors = np.load(val_errors_filename)
    raw_train_iters = np.load(train_iters_filename)
    val_iters = np.load(val_iters_filename)
    pct_pos_val = float(val_errors[0])
    if os.path.exists(pct_pos_val_filename):
        pct_pos_val = 100.0 * np.load(pct_pos_val_filename)
    raw_train_losses = np.load(train_losses_filename)

    # window the training error
    i = 0
    train_errors = []
    train_losses = []
    train_iters = []
    while i < raw_train_errors.shape[0]:
        train_errors.append(np.mean(raw_train_errors[i:i+WINDOW]))
        train_losses.append(np.mean(raw_train_losses[i:i+WINDOW]))
        train_iters.append(i)
        i += WINDOW
    train_errors = np.array(train_errors)
    train_losses = np.array(train_losses)
    train_iters = np.array(train_iters)
        
    init_val_error = val_errors[0]
    norm_train_errors = train_errors / init_val_error
    norm_val_errors = val_errors / init_val_error

    print 'Original val error', pct_pos_val
    print 'Final val error', val_errors[-1]
    print 'Normalized final val error', val_errors[-1] / pct_pos_val

    plt.figure()
    plt.plot(train_iters, train_errors, linewidth=4, color='b')
    plt.plot(val_iters, val_errors, linewidth=4, color='g')
    plt.ylim(0, 100)
    plt.legend(('Training (Minibatch)', 'Validation'), fontsize=15, loc='best')
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Error Rate', fontsize=15)
 
    plt.figure()
    plt.plot(train_iters, norm_train_errors, linewidth=4, color='b')
    plt.plot(val_iters, norm_val_errors, linewidth=4, color='g')
    plt.ylim(0, 2.0)
    plt.legend(('Training (Minibatch)', 'Validation'), fontsize=15, loc='best')
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Normalized Error Rate', fontsize=15)

    train_losses[train_losses > 100.0] = 3.0
    plt.figure()
    plt.plot(train_iters, train_losses, linewidth=4, color='b')
    plt.ylim(0, 2.0)
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Training Loss', fontsize=15)
    plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(train_iters, train_errors, linewidth=4, color='b')
    plt.plot(val_iters, val_errors, linewidth=4, color='g')
    plt.ylim(0, 100)
    plt.legend(('Training (Minibatch)', 'Validation'), fontsize=15, loc='best')
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Error Rate', fontsize=15)
    plt.savefig(os.path.join(result_dir, 'training_curve.jpg'))
 
    plt.figure(figsize=(8,6))
    plt.plot(train_iters, norm_train_errors, linewidth=4, color='b')
    plt.plot(val_iters, norm_val_errors, linewidth=4, color='g')
    plt.ylim(0, 2.0)
    plt.legend(('Training (Minibatch)', 'Validation'), fontsize=15, loc='best')
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Normalized Error Rate', fontsize=15)
    plt.savefig(os.path.join(result_dir, 'normalized_training_curve.jpg'))
  
