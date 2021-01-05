FROM tensorflow/tensorflow:2.3.0

COPY dataset /golem/dataset
RUN ls -lh /golem/dataset

WORKDIR /golem/work
VOLUME /golem/work /golem/output /golem/resource
