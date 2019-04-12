FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

MAINTAINER Vishal Satish <vsatish@berkeley.edu>

# args
# work_dir must be an absolute path
ARG work_dir=/root/Workspace 

# Install apt-get deps
RUN apt-get update && apt-get install -y \
        build-essential \ 
        python-tk \ 
        python-dev \ 
        python-opengl \ 
        curl \ 
        libsm6 \ 
        libxext6 \ 
        libglib2.0-0 \
        libxrender1 
  
# Install pip (`apt-get install python-pip` causes trouble w/ networkx)
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py

# Make working directory
WORKDIR ${work_dir}

# Copy requirements.txt and install dependencies (this allows for caching)
# COPY /requirements/gpu_requirements.txt gqcnn/requirements/gpu_requirements.txt
# RUN pip install -r gqcnn/requirements/gpu_requirements.txt

# Copy the library and install it, keeping the egg outside of the library in site-packages in case we want to mount the library (overwriting it) for development
ADD docker/gqcnn.tar .

ENV PYTHONPATH ${work_dir}/gqcnn
WORKDIR /usr/local/lib/python2.7/site-packages/
#RUN python ${work_dir}/gqcnn/setup.py develop --docker
RUN python ${work_dir}/gqcnn/setup.py develop

# Post-build setup
WORKDIR ${work_dir}

