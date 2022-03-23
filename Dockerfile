# modified from https://github.com/xfbs/docker-openpcdet
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

RUN apt update
RUN apt-get install -y software-properties-common 

RUN apt-get update
RUN apt install -y python3.6 python3-pip apt-transport-https ca-certificates gnupg software-properties-common wget git ninja-build libboost-dev build-essential

RUN python3.6 -m pip install --upgrade "pip < 21.0" 
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html


# Install CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - > /etc/apt/trusted.gpg.d/kitware.gpg
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update && apt install -y cmake

# Install spconv
RUN pip3 install pccm
WORKDIR /code
RUN git clone -b v1.2.1 https://github.com/traveller59/spconv.git --recursive 
WORKDIR /code/spconv
ENV SPCONV_FORCE_BUILD_CUDA=1
RUN python3 setup.py bdist_wheel
RUN pip3 install dist/*.whl

# Install APEX
WORKDIR /code
RUN git clone --branch centerNet_mod https://github.com/yzhou377/apex.git
WORKDIR /code/apex
RUN python3.6 -m pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install LLVM 10
WORKDIR /code
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 10

# OpenPCDet dependencies fail to install unless LLVM 10 exists on the system
# and there is a llvm-config binary available, so we have to symlink it here.
RUN ln -s /usr/bin/llvm-config-10 /usr/bin/llvm-config

RUN pip3 install --upgrade pip

ARG TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5+PTX"

# Install CenterPoint
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 0
COPY ./requirements.txt /code/requirements_install.txt
RUN pip3 install -r requirements_install.txt
RUN pip3 uninstall opencv-python  --yes
RUN pip3 install opencv-python-headless 
RUN rm /code/requirements_install.txt

RUN python3.6 -m pip install waymo-open-dataset-tf-1-15-0==1.2.0 

RUN chmod -R +777 /code 

WORKDIR /code/CenterPoint
ENV PYTHONPATH "${PYTHONPATH}:/code/CenterPoint"