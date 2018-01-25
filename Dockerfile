FROM bamos/ubuntu-opencv-dlib-torch:ubuntu_14.04-opencv_2.4.11-dlib_19.0-torch_2016.07.12

RUN apt-get update && apt-get install -y \
    curl \
    git \
    graphicsmagick \
    libssl-dev \
    libffi-dev \
    python-dev \
    python-pip \
    python-numpy \
    python-nose \
    python-scipy \
    python-pandas \
    python-protobuf \
    python-openssl \
    wget \
    zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . /root/openface
RUN python -m pip install --upgrade --force pip
RUN cd ~/openface && \
    pip2 install -r requirements.txt && \
    pip install numpy --upgrade && \
    pip install dlib --upgrade

CMD cd ~/openface/src && python2 main.py