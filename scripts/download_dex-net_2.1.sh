#!/bin/sh

wget -O data/training/dexnet_2.1.tar.gz https://berkeley.box.com/shared/static/4g0g0lstl45hv5g5232f89aoeccjk32j.zip

cd data/training
tar -xvzf dexnet_2.1.tar.gz
cd ../..
