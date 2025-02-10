import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py

import os
import os.path

import cv2

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    print('pic shape:', pic.shape)
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(image_dir, vid):
  cap = cv2.VideoCapture(os.path.join(image_dir, vid)) # might need an update here
  frames = []
  while True:
      ret, frame = cap.read()
      if not ret:
          break
      w,h,c = frame.shape
      if w < 226 or h < 226:
          d = 226.-min(w,h)
          sc = 1+d/min(w,h)
          frame = cv2.resize(frame,dsize=(0,0),fx=sc,fy=sc)
      frame = (frame/255.)*2 - 1 # normalize
      frames.append(frame)
  cap.release()
  return np.asarray(frames, dtype=np.float32)


def load_flow_frames(flow_dir, vid):
  return None

def get_num_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length

def make_dataset(root, mode, num_classes=157):
    # list files in the directory root and store them in a list that has mp4 or avi
    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f)) and (f.endswith('.mp4') or f.endswith('.avi'))]

    dataset = []
    fps = 30 # hardcoded fps
    # now open files one by one and get the frames
    for vid in files:
        label = 1 # Just one for now, you can use filenames to get the labels
        num_frames = get_num_frames(os.path.join(root, vid))
        dataset.append((vid, label, num_frames/fps, num_frames))    
    return dataset


class FolderDataset(data_utl.Dataset):

    def __init__(self, root, mode, transforms=None, save_dir='', num=0):
        
        self.data = make_dataset(root, mode)
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir

    def __getitem__(self, index):
        print('Mode is:', self.mode)
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, dur, nf = self.data[index]
        if os.path.exists(os.path.join(self.root, vid+'.npy')):
            return 0, 0, vid

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid)
        else:
            imgs = load_flow_frames(self.root, vid)

        #imgs = self.transforms(imgs)
        label = torch.ones(nf)
        return video_to_tensor(imgs), torch.from_numpy(label), vid

    def __len__(self):
        return len(self.data)
