#!/bin/bash

python extract_features.py \
  --mode rgb \
  --load_model models/rgb_imagenet.pt \
  --input_dir /abyss/home/datasets/animal_kingdom/action_recognition/dataset/subset \
  --output_dir /abyss/home/datasets/animal_kingdom/action_recognition/dataset/features/ \
  --sample_mode resize \
  --frequency 16 \
  --batch_size 64 \
  --no-usezip
