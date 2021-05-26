FROM python:3.7

EXPOSE 8083

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update -y && apt-get install -y software-properties-common

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 \
    libgl1-mesa-glx \
    libcudnn8=8.1.1.*-1+cuda11.2 \
    libcudnn8-dev=8.1.1.*-1+cuda11.2 \
    && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY ./requirements.txt /api/requirements.txt
RUN pip install -r /api/requirements.txt

COPY ./models.zip /api/models.zip
RUN unzip /api/models.zip -d /api
RUN rm /api/models.zip
RUN ls

COPY ./application/ /api/application/
COPY server.py /api/server.py
COPY ./config.yml /api/config.yml

WORKDIR /api/
CMD [ "python", "/api/server.py" ]
