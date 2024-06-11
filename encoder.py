import torch 
import torch.nn as nn
from torch.nn.functional import relu

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # output: 570x570x64
    self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

    # input: 284x284x64
    self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
    self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

    # input: 140x140x128
    self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
    self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

    # input: 68x68x256
    self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
    self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

    # input: 32x32x512
    self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
    self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024


  def forward(self, x):
    # Encoder
    xe11 = relu(self.e11(x))
    xe12 = relu(self.e12(xe11))
    xp1 = self.pool1(xe12)

    xe21 = relu(self.e21(xp1))
    xe22 = relu(self.e22(xe21))
    xp2 = self.pool2(xe22)

    xe31 = relu(self.e31(xp2))
    xe32 = relu(self.e32(xe31))
    xp3 = self.pool3(xe32)

    xe41 = relu(self.e41(xp3))
    xe42 = relu(self.e42(xe41))
    xp4 = self.pool4(xe42)

    xe51 = relu(self.e51(xp4))
    xe52 = relu(self.e52(xe51))

    return xe12, xe22, xe32, xe42, xe52