import torch
import torch.nn as nn
import torch.nn.functional as F 

# In MSBDN They add reflection padding (instead of zero padding)
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        return self.conv2d(out)



class ResBlock(nn.Module):
    """Residual Block used throughout the network."""
    def __init__(self, channels, kernel_size = 3, res_scale = 0.1):
        super(ResBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride = 1)
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride = 1)
        self.relu = nn.PReLU()
        self.res_scale = res_scale

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * self.res_scale   # Residual scaling for scability

        return out + residual


class ConvBlock_Util(nn.Module):
    def __init__(self, input_size, output_size,
                         kernel_size=3, stride=1, padding=1, 
                         bias=True, activation='prelu', norm=None):
        super(ConvBlock_Util, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        out = self.bn(self.conv(x)) if self.norm else self.conv(x)
        return self.act(out) if self.activation != 'no' else out


class DeconvBlock_Util(nn.Module):
    def __init__(self, input_size, output_size, 
                       kernel_size=4, stride=2, padding=1, 
                       bias=True, activation='prelu', norm=None):
        super(DeconvBlock_Util, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_size, output_size, kernel_size, 
                                         stride, padding, bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()

    def forward(self, x):
        out = self.bn(self.deconv(x)) if self.norm else self.deconv(x)
        return self.act(out) if self.activation is not None else out


"""
Multi-Scale Boosted Block in the Encoder block

This block is used for mixing informations from different resolutions, 
which can refine the fine-grained details much better

It works by upsampling the feature from the previous level, strengthening it with the latent feature from the encoder, 
passing it through a trainable refinement unit (a residual group), and then subtracting the original upsampled feature. 
This progressively refines the result level by level.
"""

class Encoder_MDCBlock(nn.Module):
    """Multi-scale Dense Connection Block (MDCBlock)"""
    def __init__(self, num_filter, num_ft, mode = 'iter2'):
        super(Encoder_MDCBlock, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()

        for i in range(self.num_ft):
            self.up_convs.append(
                DeconvBlock_Util(num_filter // (2 ** i), num_filter // (2 ** (i + 1)), 4, 2, 1)
            )
            self.down_convs.append(
                ConvBlock_Util(num_filter // (2 ** (i + 1)), num_filter // (2 ** i), 4, 2, 1)
            )

    def forward(self, ft_l, ft_h_list):
        # print(self)
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_l_list = []
            for i in range(len(ft_h_list)):
                ft_l_list.append(ft_l)
                ft_l = self.up_convs[self.num_ft - len(ft_h_list) + i](ft_l)

            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft_fusion = self.down_convs[self.num_ft - i - 1](ft_fusion - ft_h_list[i]) + \
                                ft_l_list[len(ft_h_list) - i - 1]

        
        if self.mode == 'iter2':
            ft_fusion = ft_l 
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)

                if ft.size() != ft_h_list[i].size():
                    target_size = ft_h_list[i].size()[2:]
                    ft = F.interpolate(ft, size = target_size, 
                                       mode = "bilinear", 
                                       align_corners = False)

                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)

                if ft.size() != ft_fusion.size():
                    target_size = ft_fusion.size()[2:]
                    ft = F.interpolate(ft, size = target_size,
                                      mode = 'bilinear',
                                      align_corners = False)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(i + 1):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[len(ft_h_list) - i - 1]
                for j in range(i+1):
                    # print(j)
                    ft = self.down_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_l
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    # print(j)
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft
                
        return ft_fusion


class Decoder_MDCBlock(nn.Module):
    def __init__(self, num_filter, num_ft, mode = 'iter2'):
        super(Decoder_MDCBlock, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()

        for i in range(self.num_ft):
            self.down_convs.append(
                ConvBlock_Util(num_filter * (2 ** i), num_filter * (2 ** (i + 1)), 4, 2, 1)
            )
            self.up_convs.append(
                DeconvBlock_Util(num_filter * (2 ** (i + 1)), num_filter * (2 ** i), 4, 2, 1)
            )

    def forward(self, ft_h, ft_l_list):
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_h_list = []
            for i in range(len(ft_l_list)):
                ft_h_list.append(ft_h)
                ft_h = self.down_convs[self.num_ft- len(ft_l_list) + i](ft_h)

            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft_fusion = self.up_convs[self.num_ft - i - 1](ft_fusion - ft_l_list[i]) +\
                            ft_h_list[len(ft_l_list) - i - 1]
                
        if self.mode == 'iter2':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                # print(f"Low Resolution {i} shape:", ft_l_list[i].shape)
                # print(f"High Resolution shape:", ft.shape)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(i+1):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[len(ft_l_list) - i - 1]
                for j in range(i+1):
                    # print(j)
                    ft = self.up_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_h
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion



class MSBDN(nn.Module):
    def __init__(self, res_blocks = 18):
        super(MSBDN, self).__init__()

        # Encoder 
        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
        self.dense0 = nn.Sequential(
            *[ResBlock(16) for _ in range(3)]
        )

        # Scale x2 down 
        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.fusion1 = Encoder_MDCBlock(32, 2, mode = "iter2") # Combines 16x and 32x
        self.dense1 = nn.Sequential(
            *[ResBlock(32) for _ in range(3)]
        )

        # Scale x4 down 
        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.fusion2 = Encoder_MDCBlock(64, 3, mode = "iter2") 
        self.dense2 = nn.Sequential(
            *[ResBlock(64) for _ in range(3)]
        )

        # Scale x8 down 
        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.fusion3 = Encoder_MDCBlock(128, 4, mode = "iter2") # Combines 16x and 32x
        self.dense3 = nn.Sequential(
            *[ResBlock(128) for _ in range(3)]
        )

        self.conv16x = ConvLayer(128, 256, kernel_size = 3, stride = 2) 
        self.fusion4 = Encoder_MDCBlock(256, 5, mode = "iter2")

        # Bottleneck Dehazing
        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze.add_module('res%d' % i, ResBlock(256))
        
        # Decoder with SOS Boosting logic
        self.convd16x = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dense_4 = nn.Sequential(*[ResBlock(128) for _ in range(3)])
        self.fusion_4 = Decoder_MDCBlock(128, 2, mode='iter2')

        self.convd8x = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dense_3 = nn.Sequential(*[ResBlock(64) for _ in range(3)])
        self.fusion_3 = Decoder_MDCBlock(64, 3, mode='iter2')

        self.convd4x = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dense_2 = nn.Sequential(*[ResBlock(32) for _ in range(3)])
        self.fusion_2 = Decoder_MDCBlock(32, 4, mode='iter2')

        self.convd2x = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dense_1 = nn.Sequential(*[ResBlock(16) for _ in range(3)])
        self.fusion_1 = Decoder_MDCBlock(16, 5, mode='iter2')

        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)

    def forward(self, x):
        # Encoder
        res1x = self.conv_input(x)
        feature_mem = [res1x]
        x_feat = self.dense0(res1x) + res1x

        res2x = self.conv2x(x_feat)
        res2x = self.fusion1(res2x, feature_mem)
        feature_mem.append(res2x)
        res2x = self.dense1(res2x) + res2x

        res4x = self.conv4x(res2x)
        res4x = self.fusion2(res4x, feature_mem)
        feature_mem.append(res4x)
        res4x = self.dense2(res4x) + res4x

        res8x = self.conv8x(res4x)
        res8x = self.fusion3(res8x, feature_mem)
        feature_mem.append(res8x)
        res8x = self.dense3(res8x) + res8x

        res16x = self.conv16x(res8x)
        res16x = self.fusion4(res16x, feature_mem)

        # SOS Boosting at Bottleneck
        res_dehaze = res16x
        in_ft = res16x * 2
        res16x = self.dehaze(in_ft) + in_ft - res_dehaze
        feature_mem_up = [res16x]

        # Decoder
        # Level 16x -> 8x
        res16x = self.convd16x(res16x)
        res16x = F.interpolate(res16x, size=res8x.size()[2:], mode='bilinear')
        res8x = res16x + res8x
        res8x = self.dense_4(res8x) + res8x - res16x # SOS subtraction
        res8x = self.fusion_4(res8x, feature_mem_up)
        feature_mem_up.append(res8x)

        # Level 8x -> 4x
        res8x = self.convd8x(res8x)
        res8x = F.interpolate(res8x, size=res4x.size()[2:], mode='bilinear')
        res4x = res8x + res4x
        res4x = self.dense_3(res4x) + res4x - res8x
        res4x = self.fusion_3(res4x, feature_mem_up)
        feature_mem_up.append(res4x)

        # Level 4x -> 2x
        res4x = self.convd4x(res4x)
        res4x = F.interpolate(res4x, size=res2x.size()[2:], mode='bilinear')
        res2x = res4x + res2x
        res2x = self.dense_2(res2x) + res2x - res4x
        res2x = self.fusion_2(res2x, feature_mem_up)
        feature_mem_up.append(res2x)

        # Level 2x -> 1x
        res2x = self.convd2x(res2x)
        res2x = F.interpolate(res2x, size=x_feat.size()[2:], mode='bilinear')
        x_out = res2x + x_feat
        x_out = self.dense_1(x_out) + x_out - res2x
        x_out = self.fusion_1(x_out, feature_mem_up)

        return self.conv_output(x_out) + x # Global residual