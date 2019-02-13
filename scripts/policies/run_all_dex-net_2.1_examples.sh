#!/bin/sh

set -e

echo "RUNNING EXAMPLE 1"
python examples/policy.py GQCNN-2.1 --depth_image data/examples/clutter/primesense/depth_0.npy --segmask data/examples/clutter/primesense/segmask_0.png --config_filename cfg/examples/replication/dex-net_2.1.yaml

echo "RUNNING EXAMPLE 2"
python examples/policy.py GQCNN-2.1 --depth_image data/examples/clutter/primesense/depth_1.npy --segmask data/examples/clutter/primesense/segmask_1.png --config_filename cfg/examples/replication/dex-net_2.1.yaml

echo "RUNNING EXAMPLE 3"
python examples/policy.py GQCNN-2.1 --depth_image data/examples/clutter/primesense/depth_2.npy --segmask data/examples/clutter/primesense/segmask_2.png --config_filename cfg/examples/replication/dex-net_2.1.yaml

echo "RUNNING EXAMPLE 4"
python examples/policy.py GQCNN-2.1 --depth_image data/examples/clutter/primesense/depth_3.npy --segmask data/examples/clutter/primesense/segmask_3.png --config_filename cfg/examples/replication/dex-net_2.1.yaml

echo "RUNNING EXAMPLE 5"
python examples/policy.py GQCNN-2.1 --depth_image data/examples/clutter/primesense/depth_4.npy --segmask data/examples/clutter/primesense/segmask_4.png --config_filename cfg/examples/replication/dex-net_2.1.yaml

