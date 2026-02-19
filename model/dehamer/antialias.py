import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print(f'Pad type [{pad_type}] not recognized')
    return PadLayer


class AntiAlias_Downsample(nn.Module):
    def __init__(self, channels, pad_type = 'reflect', filt_size = 3, 
                        stride = 2, pad_off = 0):
        super(AntiAlias_Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_type = pad_type

        # Asymmetric padding (round up at the top and round down at the bottom)
        # Perfect when kernel size is 2 
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride 
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels 

        # Define the binomial filter weights
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
            
        # Create a 2D filter by taking the outer product of the 1D filter
        filt = torch.tensor(a[:, None] * a[None, :], dtype = torch.float32)
        filt = filt / torch.sum(filt) # Normalize

        # Reshape to (out_channels, in_channels/groups, kH, kW) for 
        # depthwise convolution
        filt = filt.view(1, 1, filt_size, filt_size)
        filt = filt.repeat(channels, 1, 1, 1)

        # Register as a buffer so PyTorch knows these are NOT trainable parameters
        self.register_buffer('filt', filt)
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride] 
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride] 

        else:
            return F.conv2d(self.pad(inp), self.filt, stride = self.stride, groups = inp.shape[1])