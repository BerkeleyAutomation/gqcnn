#!/bin/sh

wget -O data/training/dexnet_3.tar.gz https://berkeley.box.com/shared/static/wd5s51f1n786i71t8dufckec0262za4f.gz

cd data/training
tar -xvzf dexnet_3.tar.gz
cd ../..
