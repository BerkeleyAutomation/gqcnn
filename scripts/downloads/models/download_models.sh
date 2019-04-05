#!/bin/sh
mkdir -p models
# STANDARD
wget -O models/GQCNN-2.0.zip https://berkeley.box.com/shared/static/j4k4z6077ytucxpo6wk1c5hwj47mmpux.zip
wget -O models/GQCNN-2.1.zip https://berkeley.box.com/shared/static/zr1gohe29r2dtaaq20iz0lqcbk5ub07y.zip
wget -O models/GQCNN-3.0.zip https://berkeley.box.com/shared/static/8l47knzbzffu8zb9y5u46q0g0rvtuk74.zip
wget -O models/GQCNN-4.0-PJ.zip https://berkeley.box.com/shared/static/boe4ilodi50hy5as5zun431s1bs7t97l.zip
wget -O models/GQCNN-4.0-SUCTION.zip https://berkeley.box.com/shared/static/kzg19axnflhwys9t7n6bnuqsn18zj9wy.zip

# FULLY-CONVOLUTIONAL
wget -O models/FC-GQCNN-4.0-PJ.zip https://berkeley.box.com/shared/static/d9tvdnudd7f0743gxixcn0k0jeg1ds71.zip
wget -O models/FC-GQCNN-4.0-SUCTION.zip https://berkeley.box.com/shared/static/ini7q54957u0cmaaxfihzn1i876m0ghd.zip

cd models
unzip -a GQCNN-2.0.zip
mv GQ-Image-Wise GQCNN-2.0
unzip -a GQCNN-2.1.zip
mv GQ-Bin-Picking-Eps90 GQCNN-2.1
unzip -a GQCNN-3.0.zip
mv GQ-Suction GQCNN-3.0
unzip -a GQCNN-4.0-PJ.zip
unzip -a GQCNN-4.0-SUCTION.zip
unzip -a FC-GQCNN-4.0-PJ.zip
unzip -a FC-GQCNN-4.0-SUCTION.zip
cd ..
