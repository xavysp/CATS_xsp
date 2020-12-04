#!/user/bin/python
# -*- encoding: utf-8 -*-

from torch.utils import data
import os, json
from os.path import join, basename
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
from .transform import *



def dataset_info(dataset_name, is_linux=True):
    if is_linux:

        config = {
            'BSDS': {
                'img_height': 321,
                'img_width': 481,
                'test_list': 'test_pair.lst',
                'data_dir': '/opt/dataset/BSDS',  # mean_rgb
                'yita': 0.5
            },
            'BSDS300': {
                'img_height': 321,
                'img_width': 481,
                'test_list': 'test_pair.lst',
                'data_dir': '/opt/dataset/BSDS300',  # NIR
                'yita': 0.5
            },
            'PASCAL': {
                'img_height': 375,
                'img_width': 500,
                'test_list': 'test_pair.lst',
                'data_dir': '/opt/dataset/PASCAL',  # mean_rgb
                'yita': 0.3
            },
            'CID': {
                'img_height': 512,
                'img_width': 512,
                'test_list': 'test_pair.lst',
                'data_dir': '/opt/dataset/CID',  # mean_rgb
                'yita': 0.3
            },
            'NYUD': {
                'img_height': 425,
                'img_width': 560,
                'test_list': 'test_pair.lst',
                'data_dir': '/opt/dataset/NYUD',  # mean_rgb
                'yita': 0.5
            },
            'MULTICUE': {
                'img_height': 720,
                'img_width': 1280,
                'test_list': 'test_pair.lst',
                'data_dir': '/opt/dataset/MULTICUE',  # mean_rgb
                'yita': 0.3
            },
            'BIPED': {
                'img_height': 720,
                'img_width': 1280,
                'test_list': 'test_rgb.txt',
                'train_list': 'train_rgb.txt',
                'data_dir': '/opt/dataset/BIPED',  # mean_rgb
                'yita': 0.5
            },
            'CLASSIC': {
                'img_height': 512,
                'img_width': 512,
                'test_list': None,
                'data_dir': 'data',  # mean_rgb
                'yita': 0.5
            },
            'DCD': {
                'img_height': 240,
                'img_width': 360,
                'test_list': 'test_pair.lst',
                'data_dir': '/opt/dataset/DCD',  # mean_rgb
                'yita': 0.2
            }
        }
    else:
        config = {
            'BSDS': {'img_height': 512,  # 321
                     'img_width': 512,  # 481
                     'test_list': 'test_pair.lst',
                     'data_dir': '../../dataset/BSDS',  # mean_rgb
                     'yita': 0.5},
            'BSDS300': {'img_height': 512,  # 321
                        'img_width': 512,  # 481
                        'test_list': 'test_pair.lst',
                        'data_dir': '../../dataset/BSDS300',  # NIR
                        'yita': 0.5},
            'PASCAL': {'img_height': 375,
                       'img_width': 500,
                       'test_list': 'test_pair.lst',
                       'data_dir': '/opt/dataset/PASCAL',  # mean_rgb
                       'yita': 0.3},
            'CID': {'img_height': 512,
                    'img_width': 512,
                    'test_list': 'test_pair.lst',
                    'data_dir': '../../dataset/CID',  # mean_rgb
                    'yita': 0.3},
            'NYUD': {'img_height': 425,
                     'img_width': 560,
                     'test_list': 'test_pair.lst',
                     'data_dir': '/opt/dataset/NYUD',  # mean_rgb
                     'yita': 0.5},
            'MULTICUE': {'img_height': 720,
                         'img_width': 1280,
                         'test_list': 'test_pair.lst',
                         'data_dir': '../../dataset/MULTICUE',  # mean_rgb
                         'yita': 0.3},
            'BIPED': {'img_height': 720,  # 720
                      'img_width': 1280,  # 1280
                      'test_list': 'test_pair.txt',
                      'train_list': 'train_pair.txt',
                      'data_dir': '../../dataset/BIPED',  # WIN: '../.../dataset/BIPED/edges'
                      'yita': 0.5},
            'CLASSIC': {'img_height': 512,
                        'img_width': 512,
                        'test_list': None,
                        'train_list': None,
                        'data_dir': 'data',  # mean_rgb
                        'yita': 0.5},
            'DCD': {'img_height': 240,
                    'img_width': 360,
                    'test_list': 'test_pair.lst',
                    'data_dir': '/opt/dataset/DCD',  # mean_rgb
                    'yita': 0.2}
        }
    return config[dataset_name]



def prepare_image_PIL(im):
    im = im[:,:,::-1] - np.zeros_like(im) # rgb to bgr
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im



class MyDataLoader(data.Dataset):
    """
    Dataloader
    """
    def __init__(self, root='./Data/NYUD', split='train', args=None,transform=True):
        self.root = root
        self.split = split

        self.transform = transform
        if self.split == 'train':
            self.data_name = args.train_data
            self.filelist = join(self.root, args.train_list)
        elif self.split == 'test':
            self.data_name = args.test_data
            self.filelist = join(self.root, args.test_list)
        else:
            raise ValueError("Invalid split type!")

        if self.data_name.lower()=='biped':
            with open(self.filelist, 'r') as f:
                self.filelist = json.load(f)

        else:
            with open(self.filelist, 'r') as f:
                self.filelist = f.readlines()

        # self.filelist = [line.strip() for line in filelist]

        # pre-processing
        if self.transform:
            self.trans = Compose([
                ColorJitter(
                    brightness = 0.5,
                    contrast = 0.5,
                    saturation = 0.5),
                #RandomCrop((512, 512))
                ])

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index] if self.data_name.lower() == 'biped' else \
                self.filelist[index].split()

            label = Image.open(join(self.root, lb_file))
            img = Image.open(join(self.root, img_file))

            if self.transform:
                im_lb = dict(im = img, lb = label)
                im_lb = self.trans(im_lb)
                img, label = im_lb['im'], im_lb['lb']

            img = np.array(img, dtype=np.float32)
            img = prepare_image_PIL(img)

            label = np.array(label, dtype=np.float32)

            if label.ndim == 3:
                label = np.squeeze(label[:, :, 0])
            assert label.ndim == 2

            label = label[np.newaxis, :, :]
            label[label == 0] = 0
            label[np.logical_and(label>0, label<=100)] = 2
            label[label > 100] = 1

            return img, label, basename(img_file).split('.')[0]
        else:
            img_file = self.filelist[index] if self.data_name.lower()=='biped' else self.filelist[index].rstrip()
            img = np.array(Image.open(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_PIL(img)
            return img, basename(img_file).split('.')[0]
