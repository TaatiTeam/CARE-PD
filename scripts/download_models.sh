#!/bin/bash

echo "📥 Downloading baseline models checkpoints..."

cd assets

gdown 1n-iZFKWmcy6UIQgW9YAkT4Y2DQILSIrp

# Unzip it
echo "📂 Unzipping..."
unzip Pretrained_checkpoints.zip
rm Pretrained_checkpoints.zip

echo "✅ Pretrained checkpoints are ready in assets/Pretrained_checkpoints"