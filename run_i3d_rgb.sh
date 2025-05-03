#!/bin/bash

python extract_features.py \
  --mode rgb \
  --load_model models/rgb_imagenet.pt \
  --input_dir /home/ulas/Documents/PhD/2.Codes/pytorch-i3d-feature-extraction/data/frames/ \
  --output_dir /home/ulas/Documents/PhD/2.Codes/pytorch-i3d-feature-extraction/data/features/ \
  --sample_mode resize \
  --frequency 16 \
  --batch_size 10 \
  --no-usezip
