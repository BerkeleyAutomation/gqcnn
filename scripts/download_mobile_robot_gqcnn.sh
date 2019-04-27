#!/bin/sh
wget -O cfg/examples/dex-net_4.0_hsr.yaml https://berkeley.box.com/shared/static/5dx6aux92zk3q2dsb6zqxhgtqww6qd1g.yaml

mkdir -p models
wget -O models/GQCNN-4.0-PJ-MOBILE-TEST.zip https://berkeley.box.com/shared/static/xmned8j5vsomu1lnn7swii82kckcvks2.zip

cd models
unzip -a GQCNN-4.0-PJ-MOBILE-TEST.zip
cd ..
