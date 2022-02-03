  
# modified from https://github.com/xfbs/docker-openpcdet
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
# Thiis is an image from the dockerhub published by the offical nvidiagitlab account. 
# It contains cuda 11.1.1, cudnn8, and ubuntu 18.04

RUN apt update

RUN apt install -y python3.7 python3-pip apt-transport-https ca-certificates gnupg software-properties-common wget git ninja-build libboost-dev build-essential

RUN pip3 install Pillow==6.2.1 torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html


# Install CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - > /etc/apt/trusted.gpg.d/kitware.gpg
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update && apt install -y cmake

# Install spconv
WORKDIR /code
RUN git clone -b v1.2.1 https://github.com/traveller59/spconv.git --recursive 
WORKDIR /code/spconv
ENV SPCONV_FORCE_BUILD_CUDA=1
RUN python3 setup.py bdist_wheel
RUN pip3 install dist/*.whl

# Install LLVM 10
WORKDIR /code
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 10

# OpenPCDet dependencies fail to install unless LLVM 10 exists on the system
# and there is a llvm-config binary available, so we have to symlink it here.
RUN ln -s /usr/bin/llvm-config-10 /usr/bin/llvm-config

RUN pip3 install --upgrade pip

# Install CenterPoint Requirements
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 0
COPY ./requirements.txt /code/requirements_install.txt
RUN pip3 install -r /code/requirements_install.txt
RUN pip3 uninstall opencv-python  --yes
RUN pip3 install opencv-python-headless 
RUN rm /code/requirements_install.txt

RUN chmod -R +777 /code 

ENV PYTHONPATH "${PYTHONPATH}:/code/CenterPoint"
