#!/bin/sh

wget -O data/training/dexnet_2.1.zip https://berkeley.box.com/shared/static/4g0g0lstl45hv5g5232f89aoeccjk32j.zip

cd data/training
unzip dexnet_2.1.zip
mv dexnet_2.1_eps_90 dex-net_2.1
cd ../..
