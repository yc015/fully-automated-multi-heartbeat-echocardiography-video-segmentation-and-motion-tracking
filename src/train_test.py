import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.clasfv_losses import deformation_motion_loss, motion_seg_loss, DiceLoss, categorical_dice
from src.transform_utils import generate_2dmotion_field
from src.loss_functions import huber_loss, convert_to_1hot, convert_to_1hot_tensor

import numpy as np


Tensor = torch.cuda.FloatTensor


def train(epoch, train_loader, model, optimizer):
    """ Training function for the network """
    model.train()
    epoch_loss = []
    ed_lv_dice = 0
    es_lv_dice = 0
    
    np.random.seed()
    for batch_idx, batch in enumerate(train_loader, 1):
        video_clips = torch.Tensor(batch[0])
        video_clips = video_clips.type(Tensor)
        filename, EF, es_clip_index, ed_clip_index, es_index, ed_index, es_frame, ed_frame, es_label, ed_label = batch[1]

        optimizer.zero_grad()
        # Get the motion tracking output from the motion tracking head using the feature map
        segmentation_output, motion_output = model(video_clips)
        
        loss = 0
        deform_loss = deformation_motion_loss(video_clips, motion_output)
        loss += deform_loss

        segmentation_loss = 0
        motion_loss = 0
        for i in range(video_clips.shape[0]):
            label_ed = np.expand_dims(ed_label.numpy(), 1).astype("int")
            label_es = np.expand_dims(es_label.numpy(), 1).astype("int")

            label_ed = label_ed[i]
            label_es = label_es[i]

            label_ed = np.expand_dims(label_ed, 0)
            label_es = np.expand_dims(label_es, 0)

            motion_one_output = motion_output[i].unsqueeze(0)
            segmentation_one_output = segmentation_output[i].unsqueeze(0)

            ed_one_index = ed_clip_index[i]
            es_one_index = es_clip_index[i]

            segmentation_one_loss, motion_one_loss = motion_seg_loss(label_ed, label_es, 
                                                                     ed_one_index, es_one_index, 
                                                                     motion_one_output, segmentation_one_output, 
                                                                     0, video_clips.shape[2], 
                                                                     F.binary_cross_entropy_with_logits)
            segmentation_loss += segmentation_one_loss
            motion_loss += motion_one_loss
        loss += (segmentation_loss / video_clips.shape[0])
        loss += (motion_loss / video_clips.shape[0])              
        
        ed_segmentations = torch.Tensor([]).type(Tensor)
        es_segmentations = torch.Tensor([]).type(Tensor)
        for i in range(len(ed_clip_index)):
            ed_one_index = ed_clip_index[i]
            es_one_index = es_clip_index[i]
            
            ed_seg = segmentation_output[i, :, ed_one_index].unsqueeze(0)
            ed_segmentations = torch.cat([ed_segmentations, ed_seg])
            
            es_seg = segmentation_output[i, :, es_one_index].unsqueeze(0)
            es_segmentations = torch.cat([es_segmentations, es_seg])
            
            
        ed_es_seg_loss = 0
        ed_es_seg_loss += F.binary_cross_entropy_with_logits(ed_segmentations, 
                                                             convert_to_1hot(np.expand_dims(ed_label.numpy().astype("int"), 1), 2), 
                                                             reduction="mean") 
        
        ed_es_seg_loss += F.binary_cross_entropy_with_logits(es_segmentations, 
                                                             convert_to_1hot(np.expand_dims(es_label.numpy().astype("int"), 1), 2), 
                                                             reduction="mean") 
        ed_es_seg_loss /= 2
        
        loss += ed_es_seg_loss

        loss.backward()
        
        optimizer.step()
        
        epoch_loss.append(loss.item())
        
        ed_segmentation_argmax = torch.argmax(ed_segmentations, 1).cpu().detach().numpy()
        es_segmentation_argmax = torch.argmax(es_segmentations, 1).cpu().detach().numpy()
            
        ed_lv_dice += categorical_dice(ed_segmentation_argmax, ed_label.numpy(), 1)
        es_lv_dice += categorical_dice(es_segmentation_argmax, es_label.numpy(), 1)
        
        # Printing the intermediate training statistics
        if batch_idx % 280 == 0:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(video_clips), len(train_loader) * len(video_clips),
                100. * batch_idx / len(train_loader), np.mean(epoch_loss)))

            print("ED LV: {:.3f}".format(ed_lv_dice / batch_idx))
            print("ES LV: {:.3f}".format(es_lv_dice / batch_idx))
            
            print("On a particular batch:")
            print("Deform loss: ", deform_loss)
            print("Segmentation loss: ", ed_es_seg_loss)
            print("Seg Motion loss: ", segmentation_loss / video_clips.shape[0], motion_loss / video_clips.shape[0])
    
    return epoch_loss


def test(epoch, test_loader, model, optimizer):
    model.eval()
    epoch_loss = []
    ed_lv_dice = 0
    es_lv_dice = 0
    
    for batch_idx, batch in enumerate(test_loader, 1):
        filename, EF, es_clip_index, ed_clip_index, es_index, ed_index, es_frame, ed_frame, es_label, ed_label = batch[1]
        with torch.no_grad():
            video_clips = torch.Tensor(batch[0])
            video_clips = video_clips.type(Tensor)

        # Get the motion tracking output from the motion tracking head using the feature map
        segmentation_output, motion_output = model(video_clips)
        
        loss = 0
        deform_loss = deformation_motion_loss(video_clips, motion_output)
        loss += deform_loss

        segmentation_loss = 0
        motion_loss = 0
        for i in range(video_clips.shape[0]):
            label_ed = np.expand_dims(ed_label.numpy(), 1).astype("int")
            label_es = np.expand_dims(es_label.numpy(), 1).astype("int")

            label_ed = label_ed[i]
            label_es = label_es[i]

            label_ed = np.expand_dims(label_ed, 0)
            label_es = np.expand_dims(label_es, 0)

            motion_one_output = motion_output[i].unsqueeze(0)
            segmentation_one_output = segmentation_output[i].unsqueeze(0)

            ed_one_index = ed_clip_index[i]
            es_one_index = es_clip_index[i]

            segmentation_one_loss, motion_one_loss = motion_seg_loss(label_ed, label_es, 
                                                                     ed_one_index, es_one_index, 
                                                                     motion_one_output, segmentation_one_output, 
                                                                     0, video_clips.shape[2], 
                                                                     F.binary_cross_entropy_with_logits)
            segmentation_loss += segmentation_one_loss
            motion_loss += motion_one_loss
        loss += (segmentation_loss / video_clips.shape[0])
        loss += (motion_loss / video_clips.shape[0])
        
        ed_segmentations = torch.Tensor([]).type(Tensor)
        es_segmentations = torch.Tensor([]).type(Tensor)
        for i in range(len(ed_clip_index)):
            ed_one_index = ed_clip_index[i]
            es_one_index = es_clip_index[i]
            
            ed_seg = segmentation_output[i, :, ed_one_index].unsqueeze(0)
            ed_segmentations = torch.cat([ed_segmentations, ed_seg])
            
            es_seg = segmentation_output[i, :, es_one_index].unsqueeze(0)
            es_segmentations = torch.cat([es_segmentations, es_seg])
            
            
        ed_es_seg_loss = 0
        ed_es_seg_loss += F.binary_cross_entropy_with_logits(ed_segmentations, 
                                                             convert_to_1hot(np.expand_dims(ed_label.numpy().astype("int"), 1), 2), 
                                                             reduction="mean") 
        
        ed_es_seg_loss += F.binary_cross_entropy_with_logits(es_segmentations, 
                                                             convert_to_1hot(np.expand_dims(es_label.numpy().astype("int"), 1), 2), 
                                                             reduction="mean") 
        ed_es_seg_loss /= 2
        
        loss += ed_es_seg_loss
        
        epoch_loss.append(loss.item())
        
        ed_segmentation_argmax = torch.argmax(ed_segmentations, 1).cpu().detach().numpy()
        es_segmentation_argmax = torch.argmax(es_segmentations, 1).cpu().detach().numpy()
        
        ed_lv_dice += categorical_dice(ed_segmentation_argmax, ed_label.numpy(), 1)
        es_lv_dice += categorical_dice(es_segmentation_argmax, es_label.numpy(), 1)
    
    print("-" * 30 + "Validation" + "-" * 30)
    print("\nED LV: {:.3f}".format(ed_lv_dice / batch_idx))
    print("ES LV: {:.3f}".format(es_lv_dice / batch_idx))
        
        # Printing the intermediate training statistics
        
    print('\nValid set: Average loss: {:.4f}\n'.format(np.mean(epoch_loss)))
    
    return epoch_loss