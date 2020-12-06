#!/user/bin/python
# -*- encoding: utf-8 -*-

import os, sys
import platform
import numpy as np
from PIL import Image
import cv2
import shutil
import argparse
import time
import datetime
import torch
from data.data_loader import *
from models.models import Network
from models.optimizer import Optimizer
from torch.utils.data import DataLoader, sampler
from utils import Logger, Averagvalue, save_checkpoint
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
from configs import Config
from train import train
from test import test, multiscale_test


IS_LINUX = True if platform.system()=="Linux" else False
# data set up
DATASET_NAMES = [
    'BIPED',
    'BSDS',
    'BSDS300',
    'CID',
    'DCD',
    'MULTICUE', #5
    'PASCAL',
    'NYUD',
    'CLASSIC'
]

# Training settings
TRAIN_DATA = DATASET_NAMES[0]  # BIPED=0
train_info = dataset_info(TRAIN_DATA, is_linux=IS_LINUX)
train_dir = train_info['data_dir']
# ----------- test -----------
TEST_DATA = DATASET_NAMES[3]  # max 8
test_info = dataset_info(TEST_DATA, is_linux=IS_LINUX)
test_dir = test_info['data_dir']
is_testing = True
base_dir = "../../dataset" if not IS_LINUX else "/opt/dataset"
base_dir = join(base_dir,TRAIN_DATA)

parser = argparse.ArgumentParser(description='Mode Selection')
parser.add_argument('--mode', default = 'test', type = str, choices={"train", "test"}, help = "Setting models for training or testing")
parser.add_argument('--base_dir', default = base_dir , type = str, help = "Setting models for training or testing")
parser.add_argument('--checkpoint', default = 'epoch-4' , type = str, help = "Setting models for training or testing")
parser.add_argument('--train_dir', default = train_dir , type = str, help = "Setting models for training or testing")
parser.add_argument('--train_list', default = train_info["train_list"] , type = str, help = "Setting models for training or testing")
parser.add_argument('--train_data', default = TRAIN_DATA , type = str, help = "Setting models for training or testing")
parser.add_argument('--test_dir', default = test_dir , type = str, help = "Setting models for training or testing")
parser.add_argument('--test_list', default = test_info["test_list"] , type = str, help = "Setting models for training or testing")
parser.add_argument('--test_data', default = TEST_DATA, type = str, help = "Setting models for training or testing")

args = parser.parse_args()

cfg = Config(train_data = TRAIN_DATA,chkpnt=args.checkpoint)

if cfg.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR, cfg.save_pth)

os.makedirs(TMP_DIR, exist_ok=True)

def main():
    torch.manual_seed(2020)
    # model
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')
    model = Network(cfg, devi=device).to(device) # enabled for GPU or CPU usage
    print('=> Load model')


    # model = model().to(device)

    test_dataset = MyDataLoader(root=args.test_dir, split="test", args=args)

    test_loader = DataLoader(test_dataset, batch_size=1,
                        num_workers=1, drop_last=True,shuffle=False)

    if args.mode == "test":
        assert isfile(cfg.resume), "No checkpoint is found at '{}'".format(cfg.resume)
        checkpoints =torch.load(cfg.resume, map_location=device)
        model.load_state_dict(checkpoints['state_dict'])
        # model.load_state_dict(torch.load(cfg.resume, map_location=device))
        test(cfg, model, test_loader, save_dir = join(TMP_DIR, "test", "sing_scl_"+args.test_data))

        if cfg.multi_aug:
            multiscale_test(model, test_loader, save_dir = join(TMP_DIR, "test", "multi_scale_test"))

    else:
        train_dataset = MyDataLoader(root=args.train_dir, split="train", transform=cfg.aug,
                                     args=args)

        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                            num_workers=1, drop_last=True,shuffle=True)

        model.init_weight()

        if cfg.resume and args.mode=='test':
            model.load_checkpoint()

        model.train()

        # optimizer
        optim, scheduler = Optimizer(cfg)(model)

        # log
        log = Logger(join(TMP_DIR, "%s-%d-log.txt" %("sgd",cfg.lr)))
        sys.stdout = log

        train_loss = []
        train_loss_detail = []

        for epoch in range(0, cfg.max_epoch):


            tr_avg_loss, tr_detail_loss = train(
                cfg,train_loader, model, optim, scheduler, epoch,
                save_dir = join(TMP_DIR, "train", "epoch-%d-training-record" % epoch),
                device=device)

            test(cfg, model, test_loader, save_dir=join(TMP_DIR, "train", "epoch-%d-testing-record-view" % epoch),
                 device=device)

            log.flush()

            train_loss.append(tr_avg_loss)
            train_loss_detail += tr_detail_loss

if __name__ == '__main__':
    main()
