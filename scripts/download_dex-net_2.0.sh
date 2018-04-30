#!/bin/sh

wget -O data/training/dexnet_2.tar.gz https://berkeley.box.com/shared/static/1b1rcnx101kaoxdrsnlzqdhpocttq9uj.gz 

cd data/training
tar -xvzf dexnet_2.tar.gz
cd ../..
