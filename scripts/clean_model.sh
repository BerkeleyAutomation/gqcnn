#!/bin/sh

START_DIR=`pwd`
cd $1
trash checkpoint
trash filters
trash learning_rates.npy
trash *.jpg
trash model_*
trash pct_pos*
trash tensorboard_summaries
trash total*
trash train*
trash val*
cd $START_DIR
