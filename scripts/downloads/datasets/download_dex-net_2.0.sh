#!/bin/sh

wget -O data/training/dexnet_2.zip https://berkeley.box.com/shared/static/15oid8m9q6n9cvr8og4vm37bwghjjslp.zip

cd data/training
unzip dexnet_2.zip
mv dexnet_2_tensor dex-net_2.0
cd ../..
