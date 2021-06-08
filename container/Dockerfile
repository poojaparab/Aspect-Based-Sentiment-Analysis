# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:18.04

MAINTAINER Amazon AI <sage-learner@amazon.com>


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip


RUN pip3 install numpy scipy scikit-learn pandas cmake xgboost flask gevent gunicorn && \
        rm -rf /root/.cache



# RUN pip3 install cmake flask gevent gunicorn
COPY requirements.txt /opt/
RUN pip3 install -r /opt/requirements.txt && rm /opt/requirements.txt
RUN python3 -m spacy download en
RUN python3 -m nltk.downloader vader_lexicon
RUN apt-get install -y git
RUN git clone https://github.com/huggingface/neuralcoref.git
RUN pip install -r neuralcoref/requirements.txt
RUN pip install -e neuralcoref/.





# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image

COPY aspect_sentiment /opt/program
WORKDIR /opt/program

