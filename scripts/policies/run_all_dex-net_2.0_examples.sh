#!/bin/sh

set -e

echo "RUNNING EXAMPLE 1"
python examples/policy.py GQCNN-2.0 --depth_image data/examples/single_object/primesense/depth_0.npy --segmask data/examples/single_object/primesense/segmask_0.png --config_filename cfg/examples/replication/dex-net_2.0.yaml

echo "RUNNING EXAMPLE 2"
python examples/policy.py GQCNN-2.0 --depth_image data/examples/single_object/primesense/depth_1.npy --segmask data/examples/single_object/primesense/segmask_1.png --config_filename cfg/examples/replication/dex-net_2.0.yaml

echo "RUNNING EXAMPLE 3"
python examples/policy.py GQCNN-2.0 --depth_image data/examples/single_object/primesense/depth_2.npy --segmask data/examples/single_object/primesense/segmask_2.png --config_filename cfg/examples/replication/dex-net_2.0.yaml

echo "RUNNING EXAMPLE 4"
python examples/policy.py GQCNN-2.0 --depth_image data/examples/single_object/primesense/depth_3.npy --segmask data/examples/single_object/primesense/segmask_3.png --config_filename cfg/examples/replication/dex-net_2.0.yaml

echo "RUNNING EXAMPLE 5"
python examples/policy.py GQCNN-2.0 --depth_image data/examples/single_object/primesense/depth_4.npy --segmask data/examples/single_object/primesense/segmask_4.png --config_filename cfg/examples/replication/dex-net_2.0.yaml

echo "RUNNING EXAMPLE 6"
python examples/policy.py GQCNN-2.0 --depth_image data/examples/single_object/primesense/depth_5.npy --segmask data/examples/single_object/primesense/segmask_5.png --config_filename cfg/examples/replication/dex-net_2.0.yaml

echo "RUNNING EXAMPLE 7"
python examples/policy.py GQCNN-2.0 --depth_image data/examples/single_object/primesense/depth_6.npy --segmask data/examples/single_object/primesense/segmask_6.png --config_filename cfg/examples/replication/dex-net_2.0.yaml

echo "RUNNING EXAMPLE 8"
python examples/policy.py GQCNN-2.0 --depth_image data/examples/single_object/primesense/depth_7.npy --segmask data/examples/single_object/primesense/segmask_7.png --config_filename cfg/examples/replication/dex-net_2.0.yaml

echo "RUNNING EXAMPLE 9"
python examples/policy.py GQCNN-2.0 --depth_image data/examples/single_object/primesense/depth_8.npy --segmask data/examples/single_object/primesense/segmask_8.png --config_filename cfg/examples/replication/dex-net_2.0.yaml

echo "RUNNING EXAMPLE 10"
python examples/policy.py GQCNN-2.0 --depth_image data/examples/single_object/primesense/depth_9.npy --segmask data/examples/single_object/primesense/segmask_9.png --config_filename cfg/examples/replication/dex-net_2.0.yaml

