import os, sys, glob, argparse
import re
import torch.utils.data
from datasets import DataSet1
# import argparse
import torchvision.transforms as transforms
from models import AutoEncoderTest
import torchvision.utils as tv_utils
from PIL import Image
from add_Functions import PSNR
from add_Functions import JPEGcompression
import numpy as np
import matplotlib.pyplot as plt

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

f = args.root_chkps + '/c_1.pth'
chkp = torch.load(f)
model.load_state_dict(chkp['model_state'])
model.to(args.device)
model.eval()

transforms_test = transforms.Compose([
    # transforms.RandomCrop((args.Image_size, args.Image_size)),
    transforms.Resize((args.Image_size, args.Image_size)),
    transforms.ToTensor()
])

dataset_test = DataSet1(args.data, mode='Testing', transforms_=transforms_test)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0)

psnr_Arr = np.zeros((2, len(dataset_test)))
with torch.no_grad():
    for i, data in enumerate(data_loader_test):
        data = data.to(args.device).type(args.dtype_float)
        output = model(data)
        Num_of_Pixels = output[1].shape[2] * output[1].shape[3]
        Num_of_Symbols = output[0].shape[1] * output[0].shape[2]
        # Num_of_Bits = Num_of_Symbols * B
        # Code_Rate = len(output[0])/(data[i].shape()[0]*data[i].shape()[1])  # Bits per pixel for the coded image

        # print(output[0].shape)
        # print(output[1].shape)

        psnr_AE = PSNR(data, output[1])
        print(psnr_AE)
        psnr_Arr[0][i] = psnr_AE

        # data_JPEG = JPEGcompression(transforms.functional.to_pil_image(data.squeeze(0)), 1)
        # psnr_JPEG = PSNR(data, transforms.ToTensor()(data_JPEG).unsqueeze_(0))
        data_JPEG = JPEGcompression(data.squeeze(0), 1)
        psnr_JPEG = PSNR(data, data_JPEG)

        print(psnr_JPEG)
        psnr_Arr[1][i] = psnr_JPEG

        # print('')
    plt.figure()
    plt.plot(range(len(dataset_test)), psnr_Arr[0][:], range(len(dataset_test)), psnr_Arr[1][:])
    plt.title("PSNR over test samples")
    plt.xlabel("tested Data Samples")
    plt.ylabel("PSNR [dB]")
    plt.grid()
    plt.legend(['AE PSNR', 'JPEG PSNR'])
    plt.show()
    plt.savefig(r'./Test_Results/PSNR_over_Test_samples')

    psnr_AE_AVG = np.mean(psnr_Arr[0])
    psnr_JPEG_AVG = np.mean(psnr_Arr[1])
    print('Average AE PSNR = ' + str(psnr_AE_AVG))
    print('Average JPEG PSNR = ' + str(psnr_JPEG_AVG))

