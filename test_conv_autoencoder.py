import os, sys, glob, argparse
import re
import torch.utils.data
from datasets import DataSet1
# import argparse
import torchvision.transforms as transforms
from models import AutoEncoderTest
import torchvision.utils as tv_utils
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=r'./dataSet/TestData', help='root directory of the dataset')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--root-chkps', type=str, default='./checkpoints', help='checkpoints directory')
parser.add_argument('--Image_size', type=int, default=128, help='size of the input images')
args = parser.parse_args()
args.device = "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
args.dtype_float = torch.cuda.FloatTensor if torch.cuda.is_available() and args.cuda else torch.FloatTensor
args.dtype_long = torch.cuda.LongTensor if torch.cuda.is_available() and args.cuda else torch.LongTensor

# get the model using our helper function
# move model to the right device
model = AutoEncoderTest(args.Image_size).to(args.device)

f = r"../checkpoints/"
chkp = torch.load(f)
model.load_state_dict(chkp['model_state'])
model.to(args.device)
model.eval()

transforms_test = transforms.ToTensor()

dataset_test = DataSet1(args.data, mode='Testing', transforms_=transforms_test)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)

with torch.no_grad():
    for i, (data, target) in enumerate(data_loader_test):
        data = data.to(args.device).type(args.dtype_float)
        output = model(data)
        Code_Rate = len(output[0])/(data[i].shape()[0]*data[i].shape()[1])  # Bits per pixel for the coded image

