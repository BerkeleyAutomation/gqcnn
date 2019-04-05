#!/bin/sh

# DOWNLOAD MODELS (if they don't exist already)
mkdir -p models
cd models

if [ ! -d "GQCNN-4.0-PJ" ]; then
    wget -O GQCNN-4.0-PJ.zip https://berkeley.box.com/shared/static/boe4ilodi50hy5as5zun431s1bs7t97l.zip
    unzip -a GQCNN-4.0-PJ.zip
else
    echo "Found existing 4.0 PJ model..."
fi

if [ ! -d "GQCNN-4.0-SUCTION" ]; then
    wget -O GQCNN-4.0-SUCTION.zip https://berkeley.box.com/shared/static/kzg19axnflhwys9t7n6bnuqsn18zj9wy.zip
    unzip -a GQCNN-4.0-SUCTION.zip
else
    echo "Found existing 4.0 suction model..."
fi

cd ..

# DOWNLOAD DATASETS (if they don't already exist)

# PARALLEL JAW
mkdir -p data/training
cd data/training

if [ ! -d "example_pj" ]; then
    wget -O example_training_dataset_pj.zip https://berkeley.box.com/shared/static/wpo8jbushrdq0adwjdsampui2tu1w1xz.zip
    unzip example_training_dataset_pj.zip
    mv grasps example_pj
else
    echo "Found existing example PJ dataset..."
fi

# SUCTION
if [ ! -d "example_suction" ]; then
    wget -O example_training_dataset_suction.zip https://berkeley.box.com/shared/static/fc9zb2cbql5rz6qtp11f6m7s0hyt1dwf.zip
    unzip example_training_dataset_suction.zip
    mv grasps example_suction
else
    echo "Found existing example suction dataset..."
fi

cd ../..

