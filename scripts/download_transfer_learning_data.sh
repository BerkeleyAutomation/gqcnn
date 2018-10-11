#!/bin/sh
mkdir -p models
wget -O models/GQCNN-4.0-SUCTION.zip https://berkeley.box.com/shared/static/kzg19axnflhwys9t7n6bnuqsn18zj9wy.zip
wget -O models/GQCNN-4.0-PJ.zip https://berkeley.box.com/shared/static/boe4ilodi50hy5as5zun431s1bs7t97l.zip

mkdir -p data
wget -O data/dex-net_4.0_pj_empirical.zip https://berkeley.box.com/shared/static/uyl0opd4ewso4np04eac6fy71wa0chhg.zip
wget -O data/dex-net_4.0_suction_empirical.zip https://berkeley.box.com/shared/static/n7l9q9jsxo8zds65o7b3ejruzx69218l.zip

cd models
unzip -a GQCNN-4.0-SUCTION.zip
unzip -a GQCNN-4.0-PJ.zip
cd ..

cd data
unzip -a dex-net_4.0_pj_empirical
mv parallel_jaw dex-net_4.0_pj_empirical
unzip -a dex-net_4.0_suction_empirical
mv suction dex-net_4.0_suction_empirical
cd ..
