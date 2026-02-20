import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, 
                 activation='prelu', norm=None):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)]
        
        if norm == 'batch':
            layers.append(nn.BatchNorm2d(out_ch))
        elif norm == 'instance':
            layers.append(nn.InstanceNorm2d(out_ch))

        acts = {
            'relu': nn.ReLU(True),
            'prelu': nn.PReLU(),
            'lrelu': nn.LeakyReLU(0.2, True),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        if activation in acts:
            layers.append(acts[activation])
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)



class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True, 
                 activation='prelu', norm=None):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)]
        
        if norm == 'batch':
            layers.append(nn.BatchNorm2d(out_ch))
        elif norm == 'instance':
            layers.append(nn.InstanceNorm2d(out_ch))

        acts = {
            'relu': nn.ReLU(True),
            'prelu': nn.PReLU(),
            'lrelu': nn.LeakyReLU(0.2, True),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        if activation in acts:
            layers.append(acts[activation])
            
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


"""
Unet section
"""
class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
        

class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode='upconv'):
        super().__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(in_chans, out_chans, kernel_size=1)
            )
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_chans * 2, out_chans, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        return self.conv_block(out)


class SAM(nn.Module):
    """Supervised Attention Module"""
    def __init__(self, n_feat, kernel_size = 1, bias = False):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=pad, bias=bias)
        self.conv2 = nn.Conv2d(n_feat, 3, kernel_size, padding=pad, bias=bias)
        self.conv3 = nn.Conv2d(3, n_feat, kernel_size, padding=pad, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img 

        x2 = torch.sigmoid(self.conv3(img))
        return (x1 * x2) + x, img


class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_1x1 = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self, x):
        return self.layers(x) + self.conv_1x1(x)


class ResBlock_fft_bench(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.main = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1) 
        self.mag  = nn.Conv2d(n_feat, n_feat, kernel_size = 1)
        self.pha = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        fre = torch.fft.rfft2(x, norm = "backward")
        mag, phase = torch.abs(fre), torch.angle(fre)

        mag_out = self.mag(mag)

        # Phase modulation via softmax-weighted attention
        # Apply the attention on the haze correction in the Amplitude sections
        mag_res = mag_out - mag
        weight = F.softmax(F.adaptive_avg_pool2d(mag_res, (1, 1)), dim=1)
        # I know which frequency channels needed the most amplitude correction (from weight). 
        # Phase errors are probably concentrated in those same channels. 
        # So scale the phase by that attention before trying to correct it 
        # — this tells the network where to focus.
        phase_out = self.pha(phase * weight) + phase

        # After finding the correct phase and amplitude, we can recover back via 
        # inverse fourier transform (IVF)
        real = mag_out * torch.cos(phase_out)
        imag = mag_out * torch.sin(phase_out)
        fre_out = torch.complex(real, imag)
        y = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')

        return self.main(x) + y


### MDC Blocks
class Encoder_MDCBlock(torch.nn.Module):
    """
    Refined Multi-scale Fusion for the Encoder path.
    Fuses Low-resolution features with a list of High-resolution feature maps.
    """
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, 
                 bias=True, activation='prelu', norm=None, mode='iter2'):
        super(Encoder_MDCBlock, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        
        for i in range(self.num_ft):
            # Channels decrease as resolution increases in the Encoder
            in_ch = num_filter // (2 ** i)
            out_ch = num_filter // (2 ** (i + 1))
            
            self.up_convs.append(
                DeconvBlock(in_ch, out_ch, kernel_size, stride, padding, bias, activation)
            )
            self.down_convs.append(
                ConvBlock(out_ch, in_ch, kernel_size, stride, padding, bias, activation)
            )

    def _forward_iter1(self, ft_l, ft_h_list):
        """
        Sequential back-projection logic for Encoder.
        """
        history = []
        n = len(ft_h_list)
        
        # Upward pass: Moving from Low-res input to High-res scales
        for i in range(n):
            history.append(ft_l)
            idx = max(0, self.num_ft - n + i)
            ft_l = self.up_convs[idx](ft_l)

        # Downward fusion pass: Calculate residual error in high-res space
        fusion = ft_l
        for i in range(n):
            residual = fusion - ft_h_list[i]
            idx = max(0, self.num_ft - i - 1)
            # Apply down-conv to residual and add back the historical low-res state
            fusion = self.down_convs[idx](residual) + history[n - i - 1]

        return fusion

    def _forward_iter2(self, ft_l, ft_h_list):
        """Interpolation-based feedback mode for Encoder."""
        fusion = ft_l
        n = len(ft_h_list)

        for i in range(n):
            temp = fusion 

            # 1. Project Up: Increase resolution to reach target skip-connection scale
            num_steps = self.num_ft - i
            for j in range(num_steps):
                temp = self.up_convs[j](temp)

            # 2. Align spatial size and compute error
            target_shape = ft_h_list[i].shape[-2:]
            if temp.shape[-2:] != target_shape:
                temp = F.interpolate(temp, size=target_shape, mode='bilinear', align_corners=False)
            error = temp - ft_h_list[i]

            # 3. Project error back down to low-resolution
            for j in range(num_steps):
                # Reverse idx for down-sampling
                down_idx = num_steps - j - 1
                error = self.down_convs[down_idx](error)
            
            # 4. Update the low-resolution fusion feature
            if fusion.shape[-2:] != error.shape[-2:]:
                fusion = F.interpolate(fusion, size=error.shape[-2:], mode='bilinear', align_corners=False)
            
            fusion = fusion + error
            
        return fusion 
    
    def forward(self, ft_low, ft_high_list):
        """
        ft_low: Low-resolution input feature map
        ft_high_list: List of high-resolution skip-connection features
        """
        if self.mode in ['iter1', 'conv']:
            return self._forward_iter1(ft_low, ft_high_list)
        elif self.mode == 'iter2':
            return self._forward_iter2(ft_low, ft_high_list)
        else:
            raise ValueError(f"Mode '{self.mode}' is not supported. Use 'iter1' or 'iter2'.")


class FSDGN(nn.Module):
    def __init__(self, num_in_ch=3, base_channel=16, up_mode='upconv', bias=False):
        super().__init__()
        
        # Channels: [16, 20, 28, 44, 76]
        chs = [base_channel, 20, 28, 44, 76]
        # Channels sequence for Encoder -> Bottleneck -> Decoder
        # E.g [16, 20, 28, 44, 76, 44, 28, 20, 16] 
        chs_enc_dec = chs + chs[-2::-1]

        # ------------------- STAGE 1 (Frequency / Global Branch) -------------------
        self.enc_convs = nn.ModuleList([nn.Conv2d(num_in_ch, chs[0], 3, 1, 1)])
        self.enc_convs.extend([UNetConvBlock(chs[i], chs[i + 1]) for i in range(4)])
        self.dec_convs = nn.ModuleList([UNetUpBlock(chs[i + 1], chs[i], up_mode) for i in range(4)])
        
        # Frequency and Residual Blocks
        self.res_blocks = nn.ModuleList([ResBlock(c) for c in chs_enc_dec])
        self.fft_blocks = nn.ModuleList([ResBlock_fft_bench(c) for c in chs_enc_dec])
        
        # Fusion Blocks (
        self.enc_fusions = nn.ModuleList([
            Encoder_MDCBlock(chs[1], 2), Encoder_MDCBlock(chs[2], 3),
            Encoder_MDCBlock(chs[3], 4), Encoder_MDCBlock(chs[4], 5)
        ])
        self.dec_fusions = nn.ModuleList([
            Decoder_MDCBlock(chs[3], 2, 5), Decoder_MDCBlock(chs[2], 3, 4),
            Decoder_MDCBlock(chs[1], 4, 3), Decoder_MDCBlock(chs[0], 5, 2)
        ])

        # ------------------- STAGE 2 (Spatial / Local Branch) -------------------
        self.enc_convs2 = nn.ModuleList([nn.Conv2d(num_in_ch, chs[0], 3, 1, 1)])
        self.enc_convs2.extend([UNetConvBlock(chs[i], chs[i+1]) for i in range(4)])
        self.dec_convs2 = nn.ModuleList([UNetUpBlock(chs[i+1], chs[i], up_mode) for i in range(4)])
        self.res_blocks2 = nn.ModuleList([ResBlock(c) for c in chs_enc_dec])
        
        self.enc_fusions2 = nn.ModuleList([
            Encoder_MDCBlock(chs[1], 2), Encoder_MDCBlock(chs[2], 3),
            Encoder_MDCBlock(chs[3], 4), Encoder_MDCBlock(chs[4], 5)
        ])
        self.dec_fusions2 = nn.ModuleList([
            Decoder_MDCBlock(chs[3], 2, 5), Decoder_MDCBlock(chs[2], 3, 4),
            Decoder_MDCBlock(chs[1], 4, 3), Decoder_MDCBlock(chs[0], 5, 2)
        ])

        # Cross-Stage Feature Fusion (CSFF)
        self.csff_enc = nn.ModuleList([nn.Conv2d(c, c, 1, bias=bias) for c in chs[:-1]])
        self.csff_dec = nn.ModuleList([nn.Conv2d(c, c, 1, bias=bias) for c in chs[:-1]])

        self.sam = SAM(chs[0], kernel_size=1)
        self.concat = nn.Conv2d(chs[0] * 2, chs[0], 3, padding=1)
        self.last = nn.Conv2d(chs[0], num_in_ch, kernel_size=1)

    def forward(self, x):
        identity = x
        
        # ================= STAGE 1: Frequency Guided Branch =================
        enc_feats = []
        out = self.enc_convs[0](x)
        out = self.fft_blocks[0](self.res_blocks[0](out))
        enc_feats.append(out)

        # Stage 1 Encoder 
        for i in range(4):
            out = self.enc_convs[i + 1](out)
            out = self.enc_fusions[i](out, enc_feats)
            out = self.fft_blocks[i + 1](self.res_blocks[i + 1](out))
            enc_feats.append(out)

        # Stage 1 Decoder
        dec_feats = [enc_feats[-1]]
        curr_out = enc_feats[-1]
        for i in range(4):
            idx = 3 - i # 3, 2, 1, 0 (Reversing back up the U-Net)
            # print("Curr_out: ", curr_out.shape)
            # print(f"Enc_feats at {idx}: {enc_feats[idx].shape}")
            
            curr_out = self.dec_convs[idx](curr_out, enc_feats[idx])
            
            # 5+i maps to [44, 28, 20, 16] in the chs_enc_dec list
            curr_out = self.res_blocks[5 + i](curr_out)
            curr_out = self.fft_blocks[5 + i](curr_out)
            curr_out = self.dec_fusions[i](curr_out, dec_feats)
            
            dec_feats.append(curr_out)

        sam_feats, stage1_img = self.sam(dec_feats[-1], identity)

        # ================= STAGE 2: Spatial Guided Branch =================
        enc_feats2 = []
        
        out_2 = self.enc_convs2[0](identity)
        y_concat = self.concat(torch.cat([out_2, sam_feats], dim = 1))
        
        # CSFF from Stage 1: dec_feats[-1] is the last output of Stage 1 decoder
        y = self.res_blocks2[0](y_concat)
        y = y + self.csff_enc[0](enc_feats[0]) + self.csff_dec[0](dec_feats[-1])
        enc_feats2.append(y)

        # --- Encoder Stage 2 ---
        for i in range(4):
            y = self.enc_convs2[i + 1](y)
            y = self.enc_fusions2[i](y, enc_feats2)
            y = self.res_blocks2[i + 1](y)
            if i < 3: # Apply CSFF skip connections
                y = y + self.csff_enc[i + 1](enc_feats[i + 1]) + \
                        self.csff_dec[i + 1](dec_feats[-(i + 2)])
            enc_feats2.append(y)

        # Decoder Stage 2
        dec_feats2 = [enc_feats2[ -1 ]]
        curr_y = enc_feats2[ -1 ]
        for i in range(4):
            idx = 3 - i
            curr_y = self.dec_convs2[idx](curr_y, enc_feats2[idx])
            curr_y = self.dec_fusions2[i](self.res_blocks2[5 + i](curr_y), dec_feats2)
            dec_feats2.append(curr_y)

        final_out = torch.clamp(self.last(dec_feats2[-1]), 0, 1)
        stage1_img = torch.clamp(stage1_img, 0, 1)

        return final_out, stage1_img