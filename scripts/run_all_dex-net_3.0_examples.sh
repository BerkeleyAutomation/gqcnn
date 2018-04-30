#!/bin/sh

echo "RUNNING EXAMPLE 0"
python examples/policy.py --depth_image data/examples/single_object/depth_0.npy --segmask data/examples/single_object/segmask_0.png --config_filename cfg/examples/dex-net_3.0.yaml

echo "RUNNING EXAMPLE 1"
python examples/policy.py --depth_image data/examples/single_object/depth_1.npy --segmask data/examples/single_object/segmask_1.png --config_filename cfg/examples/dex-net_3.0.yaml

echo "RUNNING EXAMPLE 2"
python examples/policy.py --depth_image data/examples/single_object/depth_2.npy --segmask data/examples/single_object/segmask_2.png --config_filename cfg/examples/dex-net_3.0.yaml

echo "RUNNING EXAMPLE 3"
python examples/policy.py --depth_image data/examples/single_object/depth_3.npy --segmask data/examples/single_object/segmask_3.png --config_filename cfg/examples/dex-net_3.0.yaml

echo "RUNNING EXAMPLE 4"
python examples/policy.py --depth_image data/examples/single_object/depth_4.npy --segmask data/examples/single_object/segmask_4.png --config_filename cfg/examples/dex-net_3.0.yaml

echo "RUNNING EXAMPLE 5"
python examples/policy.py --depth_image data/examples/single_object/depth_5.npy --segmask data/examples/single_object/segmask_5.png --config_filename cfg/examples/dex-net_3.0.yaml

echo "RUNNING EXAMPLE 6"
python examples/policy.py --depth_image data/examples/single_object/depth_6.npy --segmask data/examples/single_object/segmask_6.png --config_filename cfg/examples/dex-net_3.0.yaml

echo "RUNNING EXAMPLE 7"
python examples/policy.py --depth_image data/examples/single_object/depth_7.npy --segmask data/examples/single_object/segmask_7.png --config_filename cfg/examples/dex-net_3.0.yaml

echo "RUNNING EXAMPLE 8"
python examples/policy.py --depth_image data/examples/single_object/depth_8.npy --segmask data/examples/single_object/segmask_8.png --config_filename cfg/examples/dex-net_3.0.yaml

echo "RUNNING EXAMPLE 9"
python examples/policy.py --depth_image data/examples/single_object/depth_9.npy --segmask data/examples/single_object/segmask_9.png --config_filename cfg/examples/dex-net_3.0.yaml
