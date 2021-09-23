from PIL import Image
from datasets import DataSet1
import torchvision.transforms as transforms
import glob
from add_Functions import PSNR



FileFolder = r'./debug_output'

img_in = Image.open(r'./debug_output/orig_0.bmp')
# img_in.show()
img_out = Image.open(r'./debug_output/1.bmp')
# img_out.show()

f_in = transforms.ToTensor()(img_in).unsqueeze_(0)
f_out = transforms.ToTensor()(img_out).unsqueeze_(0)

psnr = PSNR(f_in, f_out)

print(psnr.item())




