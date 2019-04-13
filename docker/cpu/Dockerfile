FROM ubuntu:xenial

MAINTAINER Vishal Satish <vsatish@berkeley.edu>

# Args
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
        libxrender1 \
        wget \
        unzip
  
# Install pip (`apt-get install python-pip` causes trouble w/ networkx)
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py

# Make working directory
WORKDIR ${work_dir}

# Copy the library 
ADD docker/gqcnn.tar .

# This is because `python setup.py develop` skips install_requires (I think) and also because we want to explicitly use the GPU requirements
RUN pip install -r gqcnn/requirements/cpu_requirements.txt

# Install the library in editable mode because it's more versatile (in case we want to develop or if users want to modify things)
# Keep the egg outside of the library in site-packages in case we want to mount the library (overwriting it) for development with docker
ENV PYTHONPATH ${work_dir}/gqcnn
WORKDIR /usr/local/lib/python2.7/site-packages/
RUN python ${work_dir}/gqcnn/setup.py develop --docker

# Move to the base library dir
WORKDIR ${work_dir}/gqcnn

