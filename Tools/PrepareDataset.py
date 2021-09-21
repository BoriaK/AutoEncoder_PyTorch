import os
import shutil
import glob
import random
import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import torchvision.transforms as transforms

# Currently unused, croppation is done with random crop from within the training function

rootFolder = r'../dataSet'
startingDir = os.path.join(rootFolder, 'DataSet1')  # path to the original dataset
# DestinationDirRoot = os.path.join(rootFolder,
#                                   'Full')  # path to the destination folder for copied files
# DestinationDirSample = os.path.join(rootFolder,
#                                     'Sample')  # path to the destination folder for copied files
DestinationDirCropped = os.path.join(rootFolder,
                                     'Cropped')  # path to the destination folder for cropped files


def imCrop64(source_path, destination_path):
    all_files = glob.glob(os.path.join(source_path, '*.bmp'))
    for f in all_files:
        img = cv2.imread(f, 0)
        new_im = cv2.resize(img, (512, 512))  # All Images should be 512x512
        for y in range(int(new_im.size[0] / 128)):
            for x in range(int(new_im.size[1] / 128)):
                cropped = new_im[y * 128:y * 128 + 128, x * 128:x * 128 + 128]
                # cropped = new_im.crop(y * 128, y * 128 + 128, x * 128, x * 128 + 128)
                # for debug
                cv2.imshow('', cropped)
                cv2.waitKey(0)
                ####################
                name = Path(os.path.basename(f)).stem + '_y' + str(y) + '_x' + str(x) + '.jpg'
                cropped.save(os.path.join(destination_path, name))  # save all cropped images in destination
                # cv2.imwrite(os.path.join(destination_path, name), cropped)  # save all cropped images in destination
                # folder


if __name__ == "__main__":
    imCrop64(startingDir, DestinationDirCropped)

