FROM b.gcr.io/tensorflow/tensorflow:latest

ADD /trained ./
WORKDIR ./
CMD ["tensorboard", "--logdir", "./", "--host", "0.0.0.0", "--port", "6006"]
