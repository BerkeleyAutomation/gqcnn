FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

MAINTAINER Vishal Satish <vsatish@berkeley.edu>

# Args
# `work_dir` must be an absolute path.
ARG work_dir=/root/Workspace 

# Install `apt-get` deps.
RUN apt-get update && apt-get install -y \
        build-essential \ 
        python3 \ 
        python-dev \ 
        python3-dev \ 
        python-tk \ 
        python3-tk \ 
        python-opengl \ 
        curl \ 
        libsm6 \ 
        libxext6 \ 
        libglib2.0-0 \
        libxrender1 \
        wget \
        unzip
  
# Install pip (`apt-get install python-pip` causes trouble w/ networkx).
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# Make working directory.
WORKDIR ${work_dir}

# Copy the library.
ADD docker/gqcnn.tar .

# This is because `python setup.py develop` skips install_requires (I think) and also because we want to explicitly use the GPU requirements.
RUN python -m pip install -r gqcnn/requirements/p2/gpu_requirements.txt
RUN python3 -m pip install -r gqcnn/requirements/p3/gpu_requirements.txt

# Install the library in editable mode because it's more versatile (in case we want to develop or if users want to modify things)
# Keep the egg outside of the library in site-packages in case we want to mount the library (overwriting it) for development with docker
ENV PYTHONPATH ${work_dir}/gqcnn
WORKDIR /usr/local/lib/python2.7/site-packages/
RUN python ${work_dir}/gqcnn/setup.py develop --docker
WORKDIR /usr/local/lib/python3.5/site-packages/
RUN python3 ${work_dir}/gqcnn/setup.py develop --docker

# Move to the base library dir 
WORKDIR ${work_dir}/gqcnn
