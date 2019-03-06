#!/bin/sh
mkdir -p models
wget -O models/FC-GQCNN-4.0-SIEMENS.zip https://berkeley.box.com/shared/static/3oxtpp9l1sov60li46v2z2rfgebt8lra.zip

cd models
unzip -a FC-GQCNN-4.0-SIEMENS.zip
mv GQCNN-4.0-SIEMENS-V17 FC-GQCNN-4.0-SIEMENS
cd ..
