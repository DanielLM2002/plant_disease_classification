import torch.nn as nn
from torch.nn.functional import relu
import torch



class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()


    def forward(self, x):
        [xe12, xe22, xe32, xe42, xe52] = self.encoder(x)
        decoded = self.decoder(xe12, xe22, xe32, xe42, xe52)
        return decoded


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.e11 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        
        self.e12 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        
        self.e22 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        self.e32 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.e42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.e52 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        

    def forward(self, x):
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


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        self.d12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        self.d22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        

        self.upconv3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        
        self.d32 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        

        self.upconv4 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        
        self.d42 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        

        self.outconv = nn.Conv2d(8, 3, kernel_size=1)

    def forward(self, xe12, xe22, xe32, xe42, xe52):
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        out = self.outconv(xd42)

        return out
