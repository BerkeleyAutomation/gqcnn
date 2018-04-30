#!/bin/sh

echo "RUNNING EXAMPLE 0"
python examples/policy.py --depth_image data/examples/clutter/depth_0.npy --segmask data/examples/clutter/segmask_0.png --config_filename cfg/examples/dex-net_2.1.yaml

echo "RUNNING EXAMPLE 1"
python examples/policy.py --depth_image data/examples/clutter/depth_1.npy --segmask data/examples/clutter/segmask_1.png --config_filename cfg/examples/dex-net_2.1.yaml

echo "RUNNING EXAMPLE 2"
python examples/policy.py --depth_image data/examples/clutter/depth_2.npy --segmask data/examples/clutter/segmask_2.png --config_filename cfg/examples/dex-net_2.1.yaml

echo "RUNNING EXAMPLE 3"
python examples/policy.py --depth_image data/examples/clutter/depth_3.npy --segmask data/examples/clutter/segmask_3.png --config_filename cfg/examples/dex-net_2.1.yaml

echo "RUNNING EXAMPLE 4"
python examples/policy.py --depth_image data/examples/clutter/depth_4.npy --segmask data/examples/clutter/segmask_4.png --config_filename cfg/examples/dex-net_2.1.yaml
