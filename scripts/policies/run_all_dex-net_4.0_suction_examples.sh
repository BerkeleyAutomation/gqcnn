#!/bin/sh

set -e

echo "RUNNING EXAMPLE 1"
python examples/policy.py GQCNN-4.0-SUCTION --depth_image data/examples/clutter/phoxi/dex-net_4.0/depth_0.npy --segmask data/examples/clutter/phoxi/dex-net_4.0/segmask_0.png --config_filename cfg/examples/replication/dex-net_4.0_suction.yaml --camera_intr data/calib/phoxi/phoxi.intr

echo "RUNNING EXAMPLE 2"
python examples/policy.py GQCNN-4.0-SUCTION --depth_image data/examples/clutter/phoxi/dex-net_4.0/depth_1.npy --segmask data/examples/clutter/phoxi/dex-net_4.0/segmask_1.png --config_filename cfg/examples/replication/dex-net_4.0_suction.yaml --camera_intr data/calib/phoxi/phoxi.intr

echo "RUNNING EXAMPLE 3"
python examples/policy.py GQCNN-4.0-SUCTION --depth_image data/examples/clutter/phoxi/dex-net_4.0/depth_2.npy --segmask data/examples/clutter/phoxi/dex-net_4.0/segmask_2.png --config_filename cfg/examples/replication/dex-net_4.0_suction.yaml --camera_intr data/calib/phoxi/phoxi.intr

echo "RUNNING EXAMPLE 4"
python examples/policy.py GQCNN-4.0-SUCTION --depth_image data/examples/clutter/phoxi/dex-net_4.0/depth_3.npy --segmask data/examples/clutter/phoxi/dex-net_4.0/segmask_3.png --config_filename cfg/examples/replication/dex-net_4.0_suction.yaml --camera_intr data/calib/phoxi/phoxi.intr

echo "RUNNING EXAMPLE 5"
python examples/policy.py GQCNN-4.0-SUCTION --depth_image data/examples/clutter/phoxi/dex-net_4.0/depth_4.npy --segmask data/examples/clutter/phoxi/dex-net_4.0/segmask_4.png --config_filename cfg/examples/replication/dex-net_4.0_suction.yaml --camera_intr data/calib/phoxi/phoxi.intr
