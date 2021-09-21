import os
import sys
import argparse
import torch
import torch.utils.data
import torch.nn as nn
from models import AutoEncoder
from PIL import Image
from datasets import DataSet1
import torchvision.transforms as transforms

# This script is for training Auto encoder on unlabeled Images at BMP format

parser = argparse.ArgumentParser()
parser.add_argument('--resume-epoch', type=int, default=None, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=15, help='number of epochs of training')
parser.add_argument('--batch-size', type=int, default=16, help='size of the batches')
parser.add_argument('--data', type=str, default=r'./dataSet/extra-data/tagged', help='root directory of the dataset')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--root-chkps', type=str, default='./checkpoints', help='use GPU computation')
args = parser.parse_args()
args.device = "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
args.dtype_float = torch.cuda.FloatTensor if torch.cuda.is_available() and args.cuda else torch.FloatTensor
args.dtype_long = torch.cuda.LongTensor if torch.cuda.is_available() and args.cuda else torch.LongTensor

if not os.path.isdir(args.root_chkps):
    os.mkdir(args.root_chkps)


def trainAutoEncoder():
    pass


transforms_train = transforms.ToTensor()
transforms_test = transforms.ToTensor()

dataset_train = DataSet1(args.data, mode='Train')
dataset_valid = DataSet1(args.data, mode='Validation')

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
data_loader_test = torch.utils.data.DataLoader(
    dataset_valid, batch_size=1, shuffle=False, num_workers=0)
