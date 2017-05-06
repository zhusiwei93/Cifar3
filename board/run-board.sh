#!/bin/sh
cp -r /tmp/cifar3_train/ ./trained
docker build -t tensorboard .
docker run -p 0.0.0.0:6006:6006 -it tensorboard
