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
Makes a new split for a TensorDataset

Author
------
Jeff Mahler
"""
import argparse
import logging
import os
import time
import os

import autolab_core.utils as utils
from autolab_core import TensorDataset

if __name__ == '__main__':
    # setup logger
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Split a training TensorDataset based on an attribute')
    parser.add_argument('dataset_dir', type=str, default=None,
                        help='path to the dataset to use for training and validation')
    parser.add_argument('split_name', type=str, default=None,
                        help='name to use for the split')
    parser.add_argument('--train_pct', type=float, default=0.8,
                        help='percent of data to use for training')
    parser.add_argument('--field_name', type=str, default=None,
                        help='name of the field to split on')
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    split_name = args.split_name
    train_pct = args.train_pct
    field_name = args.field_name

    # create split
    dataset = TensorDataset.open(dataset_dir)
    train_indices, val_indices = dataset.make_split(split_name, train_pct, field_name)
