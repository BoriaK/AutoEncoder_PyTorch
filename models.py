import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)

        return x


# N_dims = sqrt[(H/8)x(W/8)]
class unFlatten(nn.Module):
    def __init__(self, N_dims):
        super(unFlatten, self).__init__()
        self.N_dims = N_dims

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], self.N_dims, self.N_dims)

        return x


class ConvBlock1(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=(3, 3), stride=1, padding=0, dilation=1):
        super(ConvBlock1, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(c_out),
            nn.PReLU()]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class ConvBlock2(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=(3, 3), stride=2, padding=0, dilation=1):
        super(ConvBlock2, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(c_out),
            nn.PReLU()]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class DeConvBlock1(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=(3, 3), stride=1, padding=0, dilation=1):
        super(DeConvBlock1, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(c_out),
            nn.PReLU()]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class DeConvBlock2(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=(4, 4), stride=2, padding=1, dilation=1):
        super(DeConvBlock2, self).__init__()

        conv_block = [
            nn.ConvTranspose2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(c_out),
            nn.PReLU()]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class EVD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y):
        # y = x.view(x.shape[0], x.shape[1], -1)
        S = torch.bmm(y, y.transpose(2, 1)) / y.shape[-1]
        d, U = torch.linalg.eig(S)
        U = U.real
        yrot = torch.matmul(U, y)
        return yrot


# c1 = num of input channels
# ci = num of filters/features in the output of the previous layer
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.l1 = ConvBlock1(c_in=1, c_out=64, kernel_size=3, stride=1, padding=0, dilation=1)
        self.l2 = ConvBlock2(64, 32, kernel_size=3, stride=2, padding=0, dilation=1)
        self.l3 = ConvBlock1(32, 64, kernel_size=3, stride=1, padding=0, dilation=1)
        self.l4 = ConvBlock2(64, 32, kernel_size=3, stride=2, padding=0, dilation=1)
        self.l5 = ConvBlock1(32, 64, kernel_size=3, stride=1, padding=0, dilation=1)
        self.l6 = ConvBlock2(64, 32, kernel_size=3, stride=2, padding=0, dilation=1)

        self.flat = Flatten()  # Avi's function
        # self.fc1 = nn.Linear(32, latent_dim)  # Not sure it's necessary

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)

        x = self.flat(x)

        # x = self.fc1(self.flat(x))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.l1 = DeConvBlock1(c_in=32, c_out=64, kernel_size=3, stride=1, padding=0, dilation=1)
        self.l2 = DeConvBlock2(64, 32, kernel_size=4, stride=2, padding=1, dilation=1)
        self.l3 = DeConvBlock1(32, 64, kernel_size=3, stride=1, padding=0, dilation=1)
        self.l4 = DeConvBlock2(64, 32, kernel_size=4, stride=2, padding=1, dilation=1)
        self.l5 = DeConvBlock1(32, 64, kernel_size=3, stride=1, padding=0, dilation=1)
        self.l6 = DeConvBlock2(64, 32, kernel_size=4, stride=2, padding=1, dilation=1)

        self.unflat = unFlatten(8)  # Avi's function

    def forward(self, x):
        x = self.unflat(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)

        # x = self.fc1(self.flat(x))
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.rotation = EVD()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        print(x)
        x = self.rotation(x)
        print(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 1, 64, 64)
    E = Encoder()
    y = E(x)
    print(y.shape)

    D = Decoder()
    z = D(y)
    print(z.shape)

    A = AutoEncoder()
    print(A(x).shape)
