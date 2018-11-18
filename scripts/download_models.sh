#!/bin/sh
mkdir -p models
wget -O models/GQCNN-2.0.zip https://berkeley.box.com/shared/static/j4k4z6077ytucxpo6wk1c5hwj47mmpux.zip
wget -O models/GQCNN-2.1.zip https://berkeley.box.com/shared/static/zr1gohe29r2dtaaq20iz0lqcbk5ub07y.zip
wget -O models/GQCNN-3.0.zip https://berkeley.box.com/shared/static/8l47knzbzffu8zb9y5u46q0g0rvtuk74.zip

cd models
unzip -a GQCNN-2.0.zip
mv GQ-Image-Wise GQCNN-2.0
unzip -a GQCNN-2.1.zip
mv GQ-Bin-Picking-Eps90 GQCNN-2.1
unzip -a GQCNN-3.0.zip
mv GQ-Suction GQCNN-3.0
cd ..
