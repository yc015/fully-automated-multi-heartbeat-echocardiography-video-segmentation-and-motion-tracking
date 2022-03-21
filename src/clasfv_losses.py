import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transform_utils import generate_2dmotion_field
from src.loss_functions import huber_loss, convert_to_1hot, convert_to_1hot_tensor

import numpy as np


class DiceLoss(nn.Module):
    """
        Dice loss
        See here: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch?scriptVersionId=68471013&cellId=4
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

    
def deformation_motion_loss(source_videos, motion_field):
    """
        OTA loss for motion tracking on echocardiographic frames
    """
    mse_criterion = nn.MSELoss()
    mse_loss = 0
    smooth_loss = 0
    
    # Deform both forward and backward from beginning to the end of video clip 
    for index in range(source_videos.shape[2] - 1):
        forward_motion = motion_field[:, :2, index,...]
        backward_motion = motion_field[:, 2:, index + 1,...]
        
        grid_forward = generate_2dmotion_field(source_videos[:, :, index,...], forward_motion)
        grid_backward = generate_2dmotion_field(source_videos[:, :, index + 1,...], backward_motion)
        
        pred_image_forward = F.grid_sample(source_videos[:, :, index,...], grid_forward, 
                                           align_corners=False, padding_mode='border')
        pred_image_backward = F.grid_sample(source_videos[:, :, index + 1,...], grid_backward, 
                                            align_corners=False, padding_mode='border')
        
        mse_loss += mse_criterion(source_videos[:, :, index + 1,...], pred_image_forward)
        mse_loss += mse_criterion(source_videos[:, :, index,...], pred_image_backward)
        
        smooth_loss += huber_loss(forward_motion)
        smooth_loss += huber_loss(backward_motion)
    
    # Averaging the resulting loss
    return (0.005 * smooth_loss + mse_loss) / 2 / (source_videos.shape[2] - 1)


def categorical_dice(prediction, truth, k, epsilon=1e-5):
    """
        Compute the dice overlap between the predicted labels and truth
        Not a loss
    """
    # Dice overlap metric for label value k
    A = (prediction == k)
    B = (truth == k)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B) + epsilon)


def motion_seg_loss(label_ed, label_es, ed_index, es_index, motion_output, seg_softmax, 
                    start=0, end=32, seg_criterion=DiceLoss()):
    """
        SGS loss that spatially transform the true ED and true ES fully forward to the end of video
        and backward to the beginning. Then, compare the forward and backward transformed pseudo labels with
        segmentation at all frames.
    """
    flow_source = convert_to_1hot(label_ed, 2)
    loss_forward = 0
    OTS_loss = 0
    OTS_criterion = DiceLoss()
    
    # Forward from ed to the end of video
    for frame_index in range(ed_index, end - 1):
        forward_motion = motion_output[:, :2, frame_index,...]
        motion_field = generate_2dmotion_field(flow_source, forward_motion)
        next_label = F.grid_sample(flow_source, motion_field, align_corners=False, mode="bilinear", padding_mode='border')
        
        if frame_index == (es_index - 1):
            one_hot_ES = convert_to_1hot(label_es, 2)
            OTS_loss += OTS_criterion(next_label, one_hot_ES)
        else:
            loss_forward += seg_criterion(seg_softmax[:, :, frame_index + 1, ...], next_label)
        flow_source = next_label
    
    # Forward from es to the end of video
    flow_source = convert_to_1hot(label_es, 2)
    for frame_index in range(es_index, end - 1):
        forward_motion = motion_output[:, :2, frame_index,...]
        motion_field = generate_2dmotion_field(flow_source, forward_motion)
        next_label = F.grid_sample(flow_source, motion_field, align_corners=False, mode="bilinear", padding_mode='border')

        loss_forward += seg_criterion(seg_softmax[:, :, frame_index + 1, ...], next_label)
        flow_source = next_label

    flow_source = convert_to_1hot(label_es, 2)
    loss_backward = 0
    
    # Backward from es to the beginning of video
    for frame_index in range(es_index, start, -1):
        backward_motion = motion_output[:, 2:, frame_index,...]
        motion_field = generate_2dmotion_field(flow_source, backward_motion)
        next_label = F.grid_sample(flow_source, motion_field, align_corners=False, mode="bilinear", padding_mode='border')
        
        if frame_index == ed_index + 1:
            one_hot_ED = convert_to_1hot(label_ed, 2)
            OTS_loss += OTS_criterion(next_label, one_hot_ED)
        else:
            loss_backward += seg_criterion(seg_softmax[:, :, frame_index - 1, ...], next_label)
        flow_source = next_label
    
    flow_source = convert_to_1hot(label_ed, 2)
    # Backward from ed to the beginning of video
    for frame_index in range(ed_index, start, -1):
        backward_motion = motion_output[:, 2:, frame_index,...]
        motion_field = generate_2dmotion_field(flow_source, backward_motion)
        next_label = F.grid_sample(flow_source, motion_field, align_corners=False, mode="bilinear", padding_mode='border')
        
        loss_backward += seg_criterion(seg_softmax[:, :, frame_index - 1, ...], next_label)
        flow_source = next_label
        
    # Averaging the resulting dice
    flow_loss = (loss_forward + loss_backward) / ((motion_output.shape[2] - 2) * 2)
    OTS_loss = OTS_loss / 2 
    
    return flow_loss, OTS_loss
