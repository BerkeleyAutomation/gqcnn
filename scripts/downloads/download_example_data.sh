#!/bin/sh

# DOWNLOAD MODELS
mkdir -p models

wget -O models/GQCNN-4.0-PJ.zip https://berkeley.box.com/shared/static/boe4ilodi50hy5as5zun431s1bs7t97l.zip
wget -O models/GQCNN-4.0-SUCTION.zip https://berkeley.box.com/shared/static/kzg19axnflhwys9t7n6bnuqsn18zj9wy.zip

cd models
unzip -a GQCNN-4.0-PJ.zip
unzip -a GQCNN-4.0-SUCTION.zip
cd ..

# DOWNLOAD DATASETS

# PARALLEL JAW
wget -O data/training/example_training_dataset_pj.zip https://berkeley.box.com/shared/static/wpo8jbushrdq0adwjdsampui2tu1w1xz.zip

mkdir -p data/training
cd data/training
unzip example_training_dataset_pj.zip
mv grasps example_pj
cd ../..

# SUCTION
wget -O data/training/example_training_dataset_suction.zip https://berkeley.box.com/shared/static/fc9zb2cbql5rz6qtp11f6m7s0hyt1dwf.zip

cd data/training
unzip example_training_dataset_suction.zip
mv grasps example_suction
cd ../..

