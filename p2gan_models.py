import torch
from torch import nn
import torchvision

class EncoderLayer(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(EncoderLayer, self).__init__()
  
    self.layer = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, padding_mode='reflect', groups=in_channels, bias=False),
        nn.InstanceNorm2d(in_channels, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding='valid', bias=False),
        nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
        nn.ReLU()
    )
  
  def forward(self, x):
    return self.layer(x)
    
class ResidualLayer(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ResidualLayer, self).__init__()
  
    self.layer = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', groups=in_channels, bias=False),
        nn.InstanceNorm2d(in_channels, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding='valid', bias=False),
        nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
    )
  
  def forward(self, x):
    return nn.ReLU()(self.layer(x) + x)

class DecoderLayer(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DecoderLayer, self).__init__()
  
    self.layer = nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', groups=in_channels, bias=False),
        nn.InstanceNorm2d(in_channels, affine=True, track_running_stats=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding='valid', bias=False),
        nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
    )
  
  def forward(self, x, sc=None):
    if sc is not None:
      return nn.ReLU()(self.layer(x) + sc)
    else:
      return nn.ReLU()(self.layer(x))
      
class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    self.enc1 = EncoderLayer(3, 32)
    self.enc2 = EncoderLayer(32, 64)
    self.enc3 = EncoderLayer(64, 128)
    self.res = ResidualLayer(128, 128)
    self.dec1 = DecoderLayer(128, 64)
    self.dec2 = DecoderLayer(64, 32)
    self.dec3 = DecoderLayer(32, 16)
    self.out =  nn.Sequential(
        nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        nn.Tanh()
    )


  def forward(self, x):
    x1 = self.enc1(x)
    x2 = self.enc2(x1)
    x3 = self.enc3(x2)
    x4 = self.res(x3)
    x5 = self.dec1(x4, x2)
    x6 = self.dec2(x5, x1)
    x7 = self.dec3(x6)
    return self.out(x7)

class Discriminator(nn.Module):
  def __init__(self, T, kernel_size):
    super(Discriminator, self).__init__()

    self.net = nn.Sequential(
        nn.Conv2d(3, 256, kernel_size=kernel_size, stride=kernel_size, padding='valid', bias=False),
        nn.BatchNorm2d(num_features=256, momentum=0.05, affine=True, eps=1e-3),
        nn.LeakyReLU(0.2),
        nn.Conv2d(256, 512, kernel_size=kernel_size, stride=kernel_size, padding='valid', bias=False),
        nn.BatchNorm2d(num_features=512, momentum=0.05, affine=True, eps=1e-3),
        nn.LeakyReLU(0.2),
        nn.Conv2d(512, 1, kernel_size=1, stride=1, padding='valid', bias=False),
        nn.BatchNorm2d(num_features=1, momentum=0.05, affine=True, eps=1e-3),
        nn.Sigmoid(),
        nn.AvgPool2d(T)
    )
  
  def forward(self, x):
    return self.net(x)
    
class Vgg16Partial(nn.Module):
  def __init__(self):
    super(Vgg16Partial, self).__init__()

    self.net = torchvision.models.vgg16(pretrained=True).features.eval()

    for param in self.net.parameters():
      param.requires_grad = False
    
  def forward(self, x):
    return self.net[:4](x)
