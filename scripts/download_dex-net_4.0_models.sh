#!/bin/sh
mkdir -p models
wget -O models/GQCNN-4.0-PJ.zip https://berkeley.box.com/shared/static/boe4ilodi50hy5as5zun431s1bs7t97l.zip
wget -O models/GQCNN-4.0-SUCTION.zip https://berkeley.box.com/shared/static/kzg19axnflhwys9t7n6bnuqsn18zj9wy.zip

cd models
unzip -a GQCNN-4.0-PJ.zip
unzip -a GQCNN-4.0-SUCTION.zip
cd ..
