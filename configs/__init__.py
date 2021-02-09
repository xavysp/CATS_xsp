#!/user/bin/python
# -*- encoding: utf-8 -*-

from os.path import join
import torch

class Config(object):
    def __init__(self, train_data="bsds", chkpnt='biped-20'):
        self.data = train_data.lower()
        # ============== training
        self.resume = "pretrained/{}.pth".format(chkpnt)
        self.msg_iter = 500
        self.gpu = '0' if torch.cuda.device_count() != 0 else None
        self.save_pth = join("./output", self.data)
        self.pretrained = "pretrained/vgg16.pth"
        self.aug = False

        # ============== testing
        self.multi_aug = False # Produce the multi-scale results
        self.side_edge = False # Output the side edges

        # ================ dataset
        self.dataset = "./data/{}".format(self.data)

        # =============== optimizer
        self.batch_size = 1
        self.lr = 1e-6
        self.momentum = 0.9
        self.wd = 2e-4
        self.stepsize = 10
        self.gamma = 0.1
        self.max_epoch = 30
        self.itersize = 10

