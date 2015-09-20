#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

DATA=data/ilsvrc12
TOOLS=build/tools

$TOOLS/compute_image_mean /data/share/ILSVRC2012/LMDB/256x256/train \
  $DATA/imagenet_mean.binaryproto

echo "Done."
