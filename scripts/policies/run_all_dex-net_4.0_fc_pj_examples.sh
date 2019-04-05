#!/bin/sh

set -e

echo "RUNNING EXAMPLE 1"
python examples/policy.py FC-GQCNN-4.0-PJ --fully_conv --depth_image data/examples/clutter/phoxi/fcgqcnn/depth_0.npy --segmask data/examples/clutter/phoxi/fcgqcnn/segmask_0.png --config_filename cfg/examples/replication/dex-net_4.0_fc_pj.yaml --camera_intr data/calib/phoxi/phoxi.intr

echo "RUNNING EXAMPLE 2"
python examples/policy.py FC-GQCNN-4.0-PJ --fully_conv --depth_image data/examples/clutter/phoxi/fcgqcnn/depth_1.npy --segmask data/examples/clutter/phoxi/fcgqcnn/segmask_1.png --config_filename cfg/examples/replication/dex-net_4.0_fc_pj.yaml --camera_intr data/calib/phoxi/phoxi.intr

echo "RUNNING EXAMPLE 3"
python examples/policy.py FC-GQCNN-4.0-PJ --fully_conv --depth_image data/examples/clutter/phoxi/fcgqcnn/depth_2.npy --segmask data/examples/clutter/phoxi/fcgqcnn/segmask_2.png --config_filename cfg/examples/replication/dex-net_4.0_fc_pj.yaml --camera_intr data/calib/phoxi/phoxi.intr

echo "RUNNING EXAMPLE 4"
python examples/policy.py FC-GQCNN-4.0-PJ --fully_conv --depth_image data/examples/clutter/phoxi/fcgqcnn/depth_3.npy --segmask data/examples/clutter/phoxi/fcgqcnn/segmask_3.png --config_filename cfg/examples/replication/dex-net_4.0_fc_pj.yaml --camera_intr data/calib/phoxi/phoxi.intr

echo "RUNNING EXAMPLE 5"
python examples/policy.py FC-GQCNN-4.0-PJ --fully_conv --depth_image data/examples/clutter/phoxi/fcgqcnn/depth_4.npy --segmask data/examples/clutter/phoxi/fcgqcnn/segmask_4.png --config_filename cfg/examples/replication/dex-net_4.0_fc_pj.yaml --camera_intr data/calib/phoxi/phoxi.intr
