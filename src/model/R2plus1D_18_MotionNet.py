from torchvision.models.video import r2plus1d_18

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

    
class R2plus1D_18_MotionNet(nn.Module):
    def __init__(self, pretrained=True, output_channels=4):
        super(R2plus1D_18_MotionNet, self).__init__()
        self.r2plus1d_model = r2plus1d_18(pretrained=pretrained)
        self.comb_1_layer = nn.Conv3d(1024, 64, 1)
        self.comb_batch_norm_1 = nn.BatchNorm3d(64)
        self.comb_relu_1 = nn.ReLU(inplace=True)

        self.comb_2_layer = nn.Conv3d(64, 64, 1)
        self.comb_batch_norm_2 = nn.BatchNorm3d(64)
        self.comb_relu_2 = nn.ReLU(inplace=True)

        self.motion_head = nn.Conv3d(64, 4, 1)
        nn.init.normal_(self.motion_head.weight, mean=0.0, std=np.sqrt(1e-5))
        self.segmentation_head = nn.Conv3d(64, 2, 1)
        
    def forward(self, x):
        # Assume the input shape (d, h, w)
        # Features output from stem channels == 64, shape = (32, 56, 56)
        output_stem = self.r2plus1d_model.stem(x)
        # Features output from block 1 channels == 64, shape = (32, 56, 56)
        output_layer_1 = self.r2plus1d_model.layer1(output_stem)
        # Features output from block 2 channels == 128, shape = (16, 28, 28)
        output_layer_2 = self.r2plus1d_model.layer2(output_layer_1)
        # Features output from block 3 channels == 256, shape = (8, 14, 14)
        output_layer_3 = self.r2plus1d_model.layer3(output_layer_2)
        # Features output from block 4 channels == 512, shape = (4, 7, 7)
        output_layer_4 = self.r2plus1d_model.layer4(output_layer_3)
        
        # Upsampling 5 features output to shape of original input (32, 112, 112)
        # Stem (32, 56, 56) -> (32, 112, 112)
        up_stem = F.interpolate(output_stem, scale_factor=[1, 2, 2], mode='trilinear', align_corners=True)
        # block 1 (32, 56, 56) -> (32, 112, 112)
        up_layer_1 = F.interpolate(output_layer_1, scale_factor=[1, 2, 2], mode='trilinear', align_corners=True)
        # block 2 (16, 28, 28) -> (32, 112, 112)
        up_layer_2 = F.interpolate(output_layer_2, scale_factor=[2, 4, 4], mode='trilinear', align_corners=True)
        # block 3 (8, 14, 14) -> (32, 112, 112)
        up_layer_3 = F.interpolate(output_layer_3, scale_factor=[4, 8, 8], mode='trilinear', align_corners=True)
        # block 4 (4, 7, 7) -> (32, 112, 112)
        up_layer_4 = F.interpolate(output_layer_4, scale_factor=[8, 16, 16], mode='trilinear', align_corners=True)
        
        # Concatenate the upsampled output: 64 + 64 + 128 + 256 + 512 = 1024
        cat_output = torch.cat([up_stem, up_layer_1, up_layer_2, up_layer_3, up_layer_4], 1)
        
        # 1024 -> 64
        x = self.comb_1_layer(cat_output)
        x = self.comb_batch_norm_1(x)
        x = self.comb_relu_1(x)
        
        # 64 -> 64
        x = self.comb_2_layer(x)
        x = self.comb_batch_norm_2(x)
        x = self.comb_relu_2(x)
        
        # Segmentation output: 64 -> 2 [Background, LV]
        segmentation_output = self.segmentation_head(x)
        
        # Motion output: 64 -> 4 [Forward x, y, backward x, y]
        motion_output = self.motion_head(x)
        motion_output = torch.tanh(motion_output)
        
        return segmentation_output, motion_output
