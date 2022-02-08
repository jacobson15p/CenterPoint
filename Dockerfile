# modified from https://github.com/xfbs/docker-openpcdet
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
# This is an image from the dockerhub published by the offical nvidiagitlab account. 
# It contains cuda 10.0 cudnn7, and ubuntu 16.04
RUN apt-get update 
RUN apt-get install -y software-properties-common 
RUN add-apt-repository ppa:fkrull/deadsnakes
RUN apt-get update
RUN apt-get install -y python3.6-dev python3-pip apt-transport-https ca-certificates gnupg software-properties-common wget git ninja-build libboost-dev build-essential


RUN python3.6 -m pip install --upgrade "pip < 21.0" 
RUN python3.6 -m pip install torch==1.7.1 torchvision==0.8.2

# Install CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - > /etc/apt/trusted.gpg.d/kitware.gpg
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ xenial main'
RUN apt-get update && apt install -y cmake

# Install spconv
WORKDIR /code
RUN git clone -b v1.2.1 https://github.com/traveller59/spconv.git --recursive 
WORKDIR /code/spconv
ENV SPCONV_FORCE_BUILD_CUDA=1
RUN python3.6 setup.py bdist_wheel
RUN python3.6 -m pip install dist/*.whl

# Install LLVM 10
WORKDIR /code
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 10
# OpenPCDet dependencies fail to install unless LLVM 10 exists on the system
# and there is a llvm-config binary available, so we have to symlink it here.
RUN ln -s /usr/bin/llvm-config-10 /usr/bin/llvm-config

# Install APEX
WORKDIR /code
RUN git clone --branch centerNet_mod https://github.com/yzhou377/apex.git
WORKDIR /code/apex
RUN python3.6 -m pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

ARG TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5+PTX"

# Install CenterPoint
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 0
COPY ./requirements.txt /code/requirements_install.txt
RUN python3.6 -m pip install -r /code/requirements_install.txt
RUN python3.6 -m pip uninstall opencv-python  --yes
RUN python3.6 -m pip install opencv-python-headless 
RUN rm /code/requirements_install.txt

RUN python3.6 -m pip install waymo-open-dataset-tf-1-15-0==1.2.0 

RUN chmod -R +777 /code 

ENV PYTHONPATH "${PYTHONPATH}:/code/CenterPoint"
WORKDIR /code/CenterPoint