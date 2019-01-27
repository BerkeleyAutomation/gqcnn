#!/bin/sh
mkdir -p models
wget -O models/FC-GQCNN-4.0-PJ.zip https://berkeley.box.com/shared/static/d9tvdnudd7f0743gxixcn0k0jeg1ds71.zip
wget -O models/FC-GQCNN-4.0-SUCTION.zip https://berkeley.box.com/shared/static/ini7q54957u0cmaaxfihzn1i876m0ghd.zip

cd models
unzip -a FC-GQCNN-4.0-PJ.zip
unzip -a FC-GQCNN-4.0-SUCTION.zip
cd ..
