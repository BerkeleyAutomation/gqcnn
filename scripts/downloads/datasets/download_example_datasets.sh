#!/bin/sh

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

# FULLY-CONVOLUTIONAL PARALLEL JAW 
wget -O data/training/example_training_dataset_pj_angular.zip https://berkeley.box.com/shared/static/2u4ew5444m90waucgsor8uoijgr9dgwr.zip

cd data/training
unzip example_training_dataset_pj_angular.zip
mv grasps example_fc_pj
cd ../..
