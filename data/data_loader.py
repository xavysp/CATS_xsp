#!/user/bin/python
# -*- encoding: utf-8 -*-

from torch.utils import data
import torch
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
                'test_list': 'test_pair.txt',
                'train_list': 'train_pair.txt',
                'data_dir': '/opt/dataset/BIPED',  # mean_rgb
                'yita': 0.5
            },

            'DCD': {
                'img_height': 240,
                'img_width': 360,
                'test_list': 'test_pair.lst',
                'data_dir': '/opt/dataset/DCD',  # mean_rgb
                'yita': 0.2
            },
            'CLASSIC': {
                'img_height': 512,
                'img_width': 512,
                'test_list': None,
                'data_dir': 'classic',  # mean_rgb
                'yita': 0.5
            }
        }
    else:
        config = {
            'BSDS': {'img_height': 512,  # 321
                     'img_width': 512,  # 481
                     'train_list': 'train_pair.txt',
                     'test_list': 'test_pair.lst',
                     'data_dir': 'C:/Users/xavysp/dataset/BSDS',  # mean_rgb
                     'yita': 0.5},
            'BSDS300': {'img_height': 512,  # 321
                        'img_width': 512,  # 481
                        'test_list': 'test_pair.lst',
                        'data_dir': 'C:/Users/xavysp/dataset/BSDS300',  # NIR
                        'yita': 0.5},
            'PASCAL': {'img_height': 375,
                       'img_width': 500,
                       'test_list': 'test_pair.lst',
                       'data_dir': '/opt/dataset/PASCAL',  # mean_rgb
                       'yita': 0.3},
            'CID': {'img_height': 512,
                    'img_width': 512,
                    'test_list': 'test_pair.lst',
                    'data_dir': 'C:/Users/xavysp/dataset/CID',  # mean_rgb
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
                      'data_dir': 'C:/Users/xavysp/dataset/BIPED',  # WIN: '../.../dataset/BIPED/edges'
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
                files = f.readlines()

            files = [line.strip() for line in files]
            files = [line.split() for line in files]
            self.filelist = [line[0] for line in files]
            # pairs = [line.split() for line in files]

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
            # for BIPED
            tmp_size = label.size
            if tmp_size[0] < 400 or tmp_size[1] < 400:
                label = label.resize(400, 400)
                img = img.resize(400, 400)
            else:
                label = label.crop((0, 0, 400, 400))
                img = img.crop((0, 0, 400, 400))
            # end for BIPED

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
            label[np.logical_and(label>0, label<=73.5)] = 2
            label[label > 73.5] = 1

            return img, label, basename(img_file).split('.')[0]
        else:
            if self.data_name.lower() == 'biped':
                img_file = self.filelist[index][0]
            else:
                # img_file = self.filelist[index].rstrip()
                img_file = self.filelist[index].rstrip().split()[0]

            img = np.array(Image.open(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_PIL(img)
            return img, basename(img_file).split('.')[0]

class Data_test(data.Dataset):

	def __init__(self, root, yita=0.5,
		mean_bgr = np.array([104.00699, 116.66877, 122.67892],),
		rgb=False, scale=None,args=None):
		self.mean_bgr = mean_bgr
		self.dataset_name = args.test_data
		self.root = root
		self.lst = args.test_list
		self.yita = yita
		self.rgb = rgb
		self.scale = scale
		self.cache = {}
		self.images_name = []
		self.img_shape = []
		# self.files = np.loadtxt(lst_dir, dtype=str)
		if self.lst is not None:
			lst_dir = os.path.join(self.root, self.lst)
			with open(lst_dir, 'r') as f:
				self.files = f.readlines()
				self.files = [line.strip().split(' ') for line in self.files]

			for i in range(len(self.files)):
				folder, filename = os.path.split(self.files[i][0])
				name, ext = os.path.splitext(filename)
				self.images_name.append(name)
				self.img_shape.append(None)
		else:
			images_path = os.listdir(self.root)
			labels_path = [None for i in images_path]
			self.files = [images_path, labels_path]
			for i in range(len(self.files[0])):
				folder, filename = os.path.split(self.files[0][i])
				name, ext = os.path.splitext(filename)
				tmp_img = cv2.imread(os.path.join(self.root,self.files[0][i]))
				tmp_shape = tmp_img.shape
				self.images_name.append(name)
				self.img_shape.append(tmp_shape)


	def __len__(self):
		lenght_data = len(self.files) if self.dataset_name !='CLASSIC' else len(self.files[0])
		return lenght_data

	def __getitem__(self, index):
		data_file = self.files[index]
		# load Image
		if self.dataset_name.lower() =='biped':
			base_im_dir = self.root+'imgs/test/'
			img_file = base_im_dir  + data_file[0]
		else:
			img_file = os.path.join(self.root,data_file[0])
		# print(img_file)
		if not os.path.exists(img_file):
			img_file = img_file.replace('jpg', 'png')
		# img = Image.open(img_file)
		# img = load_image_with_cache(img_file, self.cache)
		img = cv2.imread(img_file)
		# load gt image
		if self.dataset_name.lower() =='biped':
			base_gt_dir = self.root+'edge_maps/test/'

		gt=None
		return self.transform(img, gt)

	def transform(self, img, gt):
		img = np.array(img, dtype=np.float32)
		if self.rgb:
			img = img[:, :, ::-1] # RGB->BGR
		img -= self.mean_bgr

		if gt is None:
			# img = cv2.resize(img,dsize=(1504,1504)) # just for Robert dataset 1504
			gt = np.zeros((img.shape[:2]))
			gt = torch.from_numpy(np.array([gt])).float()
		img = img.transpose((2, 0, 1))
		# img = torch.from_numpy(img.copy()).float()
		return img, gt
