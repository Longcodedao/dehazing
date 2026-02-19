import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from .swin import SwinTransformer
from .antialias import AntiAlias_Downsample
from .mirnet import MSRB


"""
Pyramid Pooling Module: Larger receptive fields
"""
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, steps):
        super(PPM, self).__init__()
        self.features = []
    
        for step in steps:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(step),
                nn.Conv2d(in_dim, reduction_dim, kernel_size = 1, bias = False),
                nn.ReLU()
            ))
        self.features = nn.ModuleList(self.features)
    
    def forward(self, x):
        B, C, H, W = x.shape
        out = [x]
    
        for f in self.features:
            feat = f(x)
            feat = F.interpolate(feat, (H, W), mode = 'bilinear', align_corners=True)
            out.append(feat)
    
        return torch.cat(out, 1)


"""
Official Dehamer implementation
"""
class Dehamer(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3, bias = False):
        super(Dehamer, self).__init__()
        self.in_chans = 3

        # --- Feature Fusion Convolutions ---
        self.conv1 = nn.Conv2d(256 * 3, 256, 3, stride=1, padding=1) 
        self.conv1_1 = nn.Conv2d(384, 256, 3, stride=1, padding=1)     
        self.conv1_2 = nn.Conv2d(384, 256, 3, stride=1, padding=1) 
        
        self.conv2 = nn.Conv2d(128 * 3, 128, 3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(192, 128, 3, stride=1, padding=1)
        self.conv2_2= nn.Conv2d(192, 128, 3, stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(64 * 3, 64, 3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(96, 64, 3, stride=1, padding=1) 
        self.conv3_2 = nn.Conv2d(96, 64, 3, stride=1, padding=1)
        
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.ReLU=nn.ReLU(inplace=True)

        self.IN_1=nn.InstanceNorm2d(64, affine=False)
        self.IN_2=nn.InstanceNorm2d(128, affine=False)
        self.IN_3=nn.InstanceNorm2d(256, affine=False)
        
        self.PPM1 = PPM(32, 8, steps = (1, 2, 3, 4))
        self.PPM2 = PPM(64, 16, steps = (1, 2, 3, 4))
        self.PPM3 = PPM(128, 32, steps = (1, 2, 3, 4))
        self.PPM4 = PPM(256, 64, steps = (1, 2, 3, 4))

        self.MSRB1 = MSRB(256, stride = 2, height = 3, width = 1, bias = bias)
        self.MSRB2 = MSRB(128, stride = 2, height = 3, width = 1, bias = bias)
        self.MSRB3 = MSRB(64, stride = 2, height = 3, width = 1, bias = bias)
        self.MSRB4 = MSRB(32, stride = 2, height = 3, width = 1, bias = bias)
        
        # Authors do not see obvious gains with more parameters
        self.swin = SwinTransformer(pretrain_img_size = 224,
                                    patch_size = 2,
                                    in_chans = self.in_chans,
                                    embed_dim = 96,
                                    depths = [2, 2, 2],
                                    num_heads= [3, 6, 12], 
                                    window_size = 7,
                                    mlp_ratio = 4.,
                                    qkv_bias = True, 
                                    qk_scale = None,
                                    drop_rate = 0.,
                                    attn_drop_rate = 0., 
                                    drop_path_rate = 0.2,
                                    norm_layer = nn.LayerNorm, 
                                    ape=False,
                                    patch_norm=True,
                                    out_indices=(0, 1, 2),
                                    frozen_stages=-1,
                                    use_checkpoint=False)

        # --- CNN Encoder Blocks ---
        self.E_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.E_block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.E_block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.E_block4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        # --- Decoder Upsample Blocks ---
        self._block4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self._block3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self._block2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self._block1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self._block0 = nn.Sequential( # FIXED: Removed rogue parenthesis
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        # --- 1. Swin Encoder ---
        swin_out = self.swin(x) # [0]: 96 chans, [1]: 192 chans, [2]: 384 chans

        # --- 2. CNN Encoder ---
        swin_input_1 = self.PPM1(self.E_block1(x))            # Output: 64
        swin_input_2 = self.PPM2(self.E_block2(swin_input_1)) # Output: 128
        swin_input_3 = self.PPM3(self.E_block3(swin_input_2)) # Output: 256
        swin_input_4 = self.PPM4(self.E_block4(swin_input_3)) # Output: 512

        # --- 3. Bottleneck to Decoder ---
        upsample_4 = self._block4(swin_input_4)               # 512 -> 256

        # Decoder + Fusion Steps from Transformers and CNN 
        # --- Decoder Step  3 (Deepest) ---
        beta_1 = self.conv1_1(swin_out[2])                    # 384 -> 256
        gamma_1 = self.conv1_2(swin_out[2])                   # 384 -> 256
        swin_input_3_refine = self.IN_3(swin_input_3) * beta_1 + gamma_1 
        
        concat_3 = torch.cat((swin_input_3, swin_input_3_refine, upsample_4), dim=1) # 256 + 256 + 256 = 768
        decoder_3 = self.ReLU(self.conv1(concat_3))           # 768 -> 256
        
        upsample_3 = self._block3(decoder_3)                  # 256 -> 128
        upsample_3 = self.MSRB2(upsample_3) 

        # --- Decoder Step 2 (Mid) ---
        # FIXED: Changed from conv1_1 to conv2_1, aligned correctly to swin_out[1]
        beta_2 = self.conv2_1(swin_out[1])                    # 192 -> 128
        gamma_2 = self.conv2_2(swin_out[1])                   # 192 -> 128
        swin_input_2_refine = self.IN_2(swin_input_2) * beta_2 + gamma_2 
        
        concat_2 = torch.cat((swin_input_2, swin_input_2_refine, upsample_3), dim=1) # 128 + 128 + 128 = 384
        decoder_2 = self.ReLU(self.conv2(concat_2))           # 384 -> 128
        
        upsample_2 = self._block2(decoder_2)                  # 128 -> 64
        upsample_2 = self.MSRB3(upsample_2)

        # --- Decoder Step 1 (Shallow) ---
        # FIXED: Changed from conv1_1 to conv3_1, aligned correctly to swin_out[0]
        beta_3 = self.conv3_1(swin_out[0])                    # 96 -> 64
        gamma_3 = self.conv3_2(swin_out[0])                   # 96 -> 64
        swin_input_1_refine = self.IN_1(swin_input_1) * beta_3 + gamma_3 
        
        concat_1 = torch.cat((swin_input_1, swin_input_1_refine, upsample_2), dim=1) # 64 + 64 + 64 = 192
        decoder_1 = self.ReLU(self.conv3(concat_1))           # 192 -> 64
        
        # --- Final Upsampling & Output ---
        upsample_1 = self._block1(decoder_1)                  # 64 -> 32
        decoder_0 = self.ReLU(self.conv4(upsample_1))         # 32 -> 32
        result = self._block0(decoder_0)                      # 32 -> 3

        return result
