import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import h5py
import IPython

NEON_DIR = '/home/vsatish/Data/dexnet/data/models/test_dump/model_fxiiviuqqs'
NEON_DATABASE_FNAME = 'data.h5'
TF_DIR = '/home/vsatish/Data/dexnet/data/models/test_dump/model_rdqjbsrymf'
TF_FNAME = 'val_errors.npy'
OUTPUT_DIR = '/home/vsatish/Data/dexnet/data/analyses/misc_analyses/neon_tf_comparison'

if __name__ == '__main__':
    with h5py.File(os.path.join(NEON_DIR, NEON_DATABASE_FNAME), 'r') as f:
        neon_val_errors = f['metrics']['Error Rate'][:] * 100
    
    num_epochs = neon_val_errors.shape[0]
    val_iters = np.arange(num_epochs)

    tf_val_errors = np.load(os.path.join(TF_DIR, TF_FNAME))
    IPython.embed()
    tf_val_errors = tf_val_errors[1:] # neon only evaluates validation error after each epoch, whereas tf does it once before beginning training    
    tf_val_errors = tf_val_errors[1:] # TODO: Find out why TF is 2 off

    print ('Final neon val error: {}, final tf val error: {}'.format(neon_val_errors[-1], tf_val_errors[-1]))

    plt.figure(figsize=(8, 6))
    plt.margins(x=0.0)
    plt.plot(val_iters, neon_val_errors, linewidth=4, color='g')
    plt.plot(val_iters, tf_val_errors, linewidth=4, color='b')
    plt.ylim(0, 40)
    plt.legend(('Neon', 'Tensorflow'), fontsize=15, loc='upper right')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Error Rate(%)', fontsize=15)
    plt.title('Validation Error Rate')
    plt.savefig(os.path.join(OUTPUT_DIR, 'validation_error_big_dnet.png'))
