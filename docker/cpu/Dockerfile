FROM ubuntu:xenial

MAINTAINER Vishal Satish <vsatish@berkeley.edu>

# Args.
# Must be an absolute path.
ARG work_dir=/root/Workspace

# Install `apt-get` deps.
RUN apt-get update && apt-get install -y \
        build-essential \
        python3 \
        python3-dev \
        python3-tk \
        python-opengl \
        curl \
        libsm6 \
        libxext6 \
        libglib2.0-0 \
        libxrender1 \
        wget \
        unzip

# Install libspatialindex (required for latest rtree).
RUN curl -L http://download.osgeo.org/libspatialindex/spatialindex-src-1.8.5.tar.gz | tar xz && \
    cd spatialindex-src-1.8.5 && \
    ./configure && \
    make && \
    make install && \
    ldconfig && \
    cd ..

# Install pip (`apt-get install python-pip` causes trouble w/ networkx).
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# Required for easy_install to find right skimage version for Python 3.5.
RUN pip3 install -U setuptools

# Make working directory.
WORKDIR ${work_dir}

# Copy the library.
ADD docker/gqcnn.tar .

# This is because `python setup.py develop` skips `install_requires` (I think).
RUN python3 -m pip install -r gqcnn/requirements/cpu_requirements.txt

# Install the library in editable mode because it's more versatile (in case we want to develop or if users want to modify things)
# Keep the egg outside of the library in site-packages in case we want to mount the library (overwriting it) for development with docker
ENV PYTHONPATH ${work_dir}/gqcnn
WORKDIR /usr/local/lib/python3.5/site-packages/
RUN python3 ${work_dir}/gqcnn/setup.py develop --docker

# Move to the top-level gqcnn package dir.
WORKDIR ${work_dir}/gqcnn
