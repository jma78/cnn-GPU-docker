FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

MAINTAINER Jing Ma <jma78@emory.edu>

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
	python-numpy \
        python-pip \
        python-scipy \
        git \
        libhdf5-dev \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
	h5py \
        sklearn \
        pandas \
        Pillow \
        && \
    python -m ipykernel.kernelspec

# Install TensorFlow GPU version.
RUN pip --no-cache-dir install \
    https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp27-none-linux_x86_64.whl

# Set up our notebook config.
# COPY jupyter_notebook_config.py /root/.jupyter/

RUN pip install git+git://github.com/Theano/Theano.git
RUN pip install keras

WORKDIR "/root"

# Copy some examples
RUN git clone git://github.com/fchollet/keras.git

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

COPY train-cnn.py /root/
COPY full_dataset_vectors.h5 /root/
COPY keras.json /root/.keras/keras.json
COPY . /root

CMD ["/bin/bash"]