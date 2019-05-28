#!/bin/bash

# Copyright ©2017. The Regents of the University of California (Regents).
# All Rights Reserved. Permission to use, copy, modify, and distribute this
# software and its documentation for educational, research, and not-for-profit
# purposes, without fee and without a signed licensing agreement, is hereby
# granted, provided that the above copyright notice, this paragraph and the
# following two paragraphs appear in all copies, modifications, and
# distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
# Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
# otl@berkeley.edu,
# http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
# THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
# HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
# MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

set -e

echo "RUNNING EXAMPLE 1"
python examples/policy.py FC-GQCNN-4.0-SUCTION --fully_conv --depth_image data/examples/clutter/phoxi/fcgqcnn/depth_0.npy --segmask data/examples/clutter/phoxi/fcgqcnn/segmask_0.png --config_filename cfg/examples/replication/dex-net_4.0_fc_suction.yaml --camera_intr data/calib/phoxi/phoxi.intr

echo "RUNNING EXAMPLE 2"
python examples/policy.py FC-GQCNN-4.0-SUCTION --fully_conv --depth_image data/examples/clutter/phoxi/fcgqcnn/depth_1.npy --segmask data/examples/clutter/phoxi/fcgqcnn/segmask_1.png --config_filename cfg/examples/replication/dex-net_4.0_fc_suction.yaml --camera_intr data/calib/phoxi/phoxi.intr

echo "RUNNING EXAMPLE 3"
python examples/policy.py FC-GQCNN-4.0-SUCTION --fully_conv --depth_image data/examples/clutter/phoxi/fcgqcnn/depth_2.npy --segmask data/examples/clutter/phoxi/fcgqcnn/segmask_2.png --config_filename cfg/examples/replication/dex-net_4.0_fc_suction.yaml --camera_intr data/calib/phoxi/phoxi.intr

echo "RUNNING EXAMPLE 4"
python examples/policy.py FC-GQCNN-4.0-SUCTION --fully_conv --depth_image data/examples/clutter/phoxi/fcgqcnn/depth_3.npy --segmask data/examples/clutter/phoxi/fcgqcnn/segmask_3.png --config_filename cfg/examples/replication/dex-net_4.0_fc_suction.yaml --camera_intr data/calib/phoxi/phoxi.intr

echo "RUNNING EXAMPLE 5"
python examples/policy.py FC-GQCNN-4.0-SUCTION --fully_conv --depth_image data/examples/clutter/phoxi/fcgqcnn/depth_4.npy --segmask data/examples/clutter/phoxi/fcgqcnn/segmask_4.png --config_filename cfg/examples/replication/dex-net_4.0_fc_suction.yaml --camera_intr data/calib/phoxi/phoxi.intr
