import torch
import numpy as np
from torch import nn
from src.transform_utils import generate_2dmotion_field
import torch.nn.functional as F

Tensor = torch.cuda.FloatTensor
# classes = ["Background", "Endo", "Epi", "LA"]
classes = ["Background", "LV", "Myo", "LA"]
CE_criterion = nn.CrossEntropyLoss()

# General notes: For shape, N is the batch size, C is the channel size, D is the dimension, H is the height, and W is the width
# Local cross correlation == cross correlation when n = 1, where n is the number of subimage (window)
# In general local cross correlation <= cross correlation for n >= 1
# dice_loss takes a segmentation output after softmax and a one-hot encoded label
# OTA is the Appearance level motion tracking loss SGA is the Appearance level segmentation loss OTS is the shape level motion tracking loss
# SGS is the shape level segmentation loss
# convert_to_1hot takes input with numpy type and all operations are on cpu. convert_to_1hot_Tensor takes input on Tensor and all operations are on Tensor

# Potential problem: weight on the dice loss. Dice loss is too small comparing with the CE loss
# For smoothness use the sum of the (dx * dx + dy * dy) then average it on the batch level


def local_cross_correlation(template, source, n=4):
    r""" The local cross correlation which subtract the template and source image with the local mean 
    in n x n windows (subimages) with size of H / n x W / n
        The template and source image should have the same shape N x C x H x W
        n: The number of intervals in Height and width
    """
    assert template.shape == source.shape, "template and source have unequal shape"
    assert template.shape[-1] % n == 0 and template.shape[-2] % n == 0,\
    "image's width and height must a be integer mutliple of number of window n"
    
    template_image = template.clone()
    source_image = source.clone()
    window_size = template.shape[-1] // n
    # Subtract the local mean from the subimage (window)
    for row_index in range(0, template.shape[-2], window_size):
        for col_index in range(0, template.shape[-1], window_size):
            template_image[:, :, row_index : row_index + window_size, col_index : col_index + window_size] -= \
            torch.mean(template_image[:, :, row_index : row_index + window_size, col_index : col_index + window_size])
            source_image[:, :, row_index : row_index + window_size, col_index : col_index + window_size] -= \
            torch.mean(source_image[:, :, row_index : row_index + window_size, col_index : col_index + window_size])
            
    # Compute the local cross correlation
    local_cross_corre = torch.sum(template_image * source_image, dim=(3, 2)) / \
    (torch.sqrt(torch.sum(template_image * template_image, dim=(3, 2))) * torch.sqrt(torch.sum(source_image * source_image, dim=(3, 2))))
    
    return torch.mean(local_cross_corre)


def cross_correlation(template, source):
    r""" The Pearson cross correlation
         template: can either be pred or target
         source: can either be pred or target 
         Expected the template and the source image to be the same shape N x C x H x W
    """
    template_image = template - torch.mean(template)
    source_image = source - torch.mean(source)
    cross_corre = torch.sum(template_image * source_image, dim=(3, 2)) / \
    (torch.sqrt(torch.sum(source_image * source_image, dim=(3, 2))) * torch.sqrt(torch.sum(template_image * template_image, dim=(3, 2))))
    
    return cross_corre


def huber_loss(x):
    # Grab from qin's main.py of joint-seg-motion-net
    bsize, csize, height, width = x.size()

    d_x = torch.index_select(x, 3, torch.arange(1, width).cuda()) - torch.index_select(x, 3, torch.arange(width-1).cuda())
    d_y = torch.index_select(x, 2, torch.arange(1, height).cuda()) - torch.index_select(x, 2, torch.arange(height-1).cuda())
    
    err = torch.sum(torch.mul(d_x, d_x)) / height + torch.sum(torch.mul(d_y, d_y)) / width
    err /= bsize
    tv_err = torch.sqrt(0.01+err)

    return tv_err


def smoothness_loss(motion_output):
    r""" The smoothness loss computed using the displacement u (motion_field)
         motion_output: the output from the motion tracking head shape=N x C(2) x H x W """
    # Mean 
    dy = motion_output[:, 0, 1:, :-1] - motion_output[:, 0, :-1, :-1]
    dx = motion_output[:, 1, :-1, 1:] - motion_output[:, 1, :-1, :-1]
    #loss = dx.abs().mean() + dy.abs().mean()
    #loss = torch.mean(torch.sum(dx ** 2 + dy ** 2, dim=(1, 2)))
    loss = torch.mean(dx ** 2 + dy ** 2)
    
    return loss


def OTA_loss(source_image, motion_field, regulator_gamma=10):
    r""" The appearance level motion tracking loss
         source_image: The echo image sequence. 
         motion_field: The output from the motion tracking head shape=N x C (4) x D (10) x H x W
         regulator_gamma: A trade-off parameter for smoothness loss
    """
    corre_loss = 0
    smooth_loss = 0
    for index in range(source_image.shape[2] - 1):
        forward_motion = motion_field[:, :2, index,...]
        #backward_motion = motion_field[:, 2:, index,...]
        backward_motion = motion_field[:, 2:, index + 1,...]
        
        grid_forward = generate_2dmotion_field(source_image[:, :, index,...], forward_motion)
        grid_backward = generate_2dmotion_field(source_image[:, :, index + 1,...], backward_motion)
        
        pred_image_forward = F.grid_sample(source_image[:, :, index,...], grid_forward, align_corners=False, mode='bilinear', padding_mode='border')
        pred_image_backward = F.grid_sample(source_image[:, :, index + 1,...], grid_backward, align_corners=False, mode='bilinear', padding_mode='border')
        
        corre_loss += local_cross_correlation(source_image[:, :, index + 1,...], pred_image_forward, 4)
        corre_loss += local_cross_correlation(source_image[:, :, index,...], pred_image_backward, 4)
        
        smooth_loss += smoothness_loss(forward_motion)
        smooth_loss += smoothness_loss(backward_motion)
        #smooth_loss += huber_loss(forward_motion)
        #smooth_loss += huber_loss(backward_motion)
    
    return (-1 / (2 * (source_image.shape[2] - 1))) * corre_loss + (regulator_gamma / (2 * (source_image.shape[2] - 1))) * smooth_loss
        
    
def convert_to_1hot(label, n_class):
    # Qin's code for Joint Motion seg learning. 
    #See: https://github.com/cq615/Joint-Learning-of-Motion-Estimation-and-Segmentation-for-Cardiac-MR-Image-Sequences/blob/e609440409f42b7b7e73ca6840eb7d89f3dd4efb/pytorch_version/main.py#L25
    # Convert a label map (N x 1 x H x W) into a one-hot representation (N x C x H x W)
    label_swap = label.swapaxes(1, 3)
    label_flat = label_swap.flatten()
    n_data = len(label_flat)
    label_1hot = np.zeros((n_data, n_class), dtype='int16')
    label_1hot[range(n_data), label_flat] = 1
    label_1hot = label_1hot.reshape((label_swap.shape[0], label_swap.shape[1], label_swap.shape[2], n_class))
    label_1hot = label_1hot.swapaxes(1, 3)
    return torch.Tensor(label_1hot).cuda()


def dice_loss(softmax_output, label, class_index=0, eps=0):
    r""" The differentiable multi-class dice loss 
         softmax_output: The output from the segmentation head after the F.softmax. shape = N x C x H x W
         label: The target label with shape = N x C x H x W
         class_index: The class index
         eps: smoothness term
     """
    assert softmax_output.shape == label.shape, "Incompatible shapes between the softmax output and label"
    assert len(softmax_output.shape) == 4, "The dimension of the softmax output has to be 4"
    assert len(label.shape) == 4, "The dimension of the label has to be 4"
    # Maybe just torch.sum(label * softmax_output, dim=(3, 2))
    # denom_soft = torch.sum(softmax_output * softmax_output, dim=(3, 2))
    # denom_label = torch.sum(label * label, dim=(3, 2))
    # this should give the average dices of each batch while we could not weight the class dice individually with this implementation
    nom = torch.sum(label[:, class_index, ...] * softmax_output[:, class_index, ...], dim=(2, 1))
    denom_soft = torch.sum(softmax_output[:, class_index,...] * softmax_output[:, class_index,...], dim=(2, 1))
    denom_label = torch.sum(label[:, class_index,...] * label[:, class_index,...], dim=(2, 1))
    
    loss = (2 * nom + eps) / (denom_soft + denom_label + eps) 
    return 1 - torch.mean(loss)


def SGA_loss(labels, seg_out, seg_softmax, omega=1, weighted=False):
    r""" The appearance level segmentation loss 
         labels: The data dictionary that contains the key "label_ED" and "label_ES" 
         which is the batch in the current implementation. 
         seg_out: The segmentation output from the segmentation head
         seg_out_max: The segmentation map get from torch.argmax(seg_out, dim=1) 
         omega: 
         weighted: True then weight the dice loss by the inverse of the class size False weight the dice loss equally
     """
    # Cross Entropy loss
    ce_loss = 0
    # Multi class dice loss
    multi_class_dice = 0
    for label_name, frame_index in zip(["label_ED", "label_ES"], [0, 9]):
        # pred is the segmented ED or ES of all batch (before softmax)
        pred = seg_out[:, :, frame_index,...]
        # pred_softmax is the pred after softmax
        pred_softmax = seg_softmax[:, :, frame_index,...]
        # label is the target ED or ES label current shape N x 1 x H x W
        label = torch.Tensor(labels[label_name]).type(torch.cuda.LongTensor)
        # Now change the label's shape to N x H x W for cross entropy computation
        label = label.view(-1, label.shape[-2], label.shape[-1])
        # One hot encoded label, labels[label_name] shape = N x 1 x H x W. One hot label shape = N x C(4) x H x W
        one_hot_label = convert_to_1hot(labels[label_name], 4)
        
        # It maybe okay to just use the class size of the whole input data instead of a batch
        weights = get_weights(label, weighted)
        # Accumulate the multi-class dice part
        for cate_index, category in enumerate(classes):
            multi_class_dice += dice_loss(pred_softmax, one_hot_label, cate_index) * (1 / weights[category])
        # Accumulate the cross entropy loss part
        ce_loss += CE_criterion(pred, label)
       
    return (1 / (2 * omega)) * (ce_loss + multi_class_dice / len(classes))


def convert_to_1hot_tensor(label, n_class):
    r""" Helper function that convert a label image with shape N x 1 x H x W into one-hot encoding N x n_class x H x W
         where all operations stay on the Tensors. See: https://discuss.pytorch.org/t/one-hot-encoding-with-autograd-dice-loss/9781/3
    """
    # Ensure the label is in the type of LongTensor 
    label = label.type(torch.cuda.LongTensor)
    # Get the place holder for the one-hot encoded label make sure it's on cuda
    one_hot_encoded_label = torch.zeros(size=(label.shape[0], n_class, label.shape[2], label.shape[3])).cuda()
    # Get the one hot encoded label
    one_hot_encoded_label.scatter_(1, label, 1)
    return one_hot_encoded_label


def SGS_OTS_loss(labels, motion_output, seg_softmax, omega=1, weighted=False):
    r""" Stage 2 loss: SGS is the Shape level segmentation loss and OTS is the Shape level motion tracking loss 
         labels: the data dictionary that contains the key "label_ED" and "label_ES" which is the batch in current implementation 
         shape = N x C x H x W
         motion_output: The output from the motion_tracking head. [N, 4 channel (fore_y, fore_x, back_y, back_x), D, H, W]
         seg_softmax: The output from the segmentation head after torch.argmax() shape = N x C x D x H x W
         omega: 
         weighted: True then weight the dice loss by the inverse of the class size False weight the dice loss equally
    """
    # Frame ED is 1
    # Frame ES is 10
    # From ED to Frame 9 Using motion field 1 - 8. Motion field 1 + frame 1 (ED) = transform frame 2 compare with pred segmented output frame 2
    # Motion field i + transform frame i = transform frame (i + 1) compare with pred segmented output frame (i + 1)
    # flow_source = torch.Tensor(labels["label_ED"]).cuda()
    flow_source = convert_to_1hot(labels["label_ED"], 4)
    loss_forward = 0
    OTS_loss = 0
    
    weights = get_weights(torch.Tensor(labels["label_ED"]), weighted)
    
    for frame_index in range(0, motion_output.shape[2] - 1):
        forward_motion = motion_output[:, :2, frame_index,...]
        motion_field = generate_2dmotion_field(flow_source, forward_motion)
        next_label = F.grid_sample(flow_source, motion_field, align_corners=False, mode="bilinear", padding_mode='border')
        
        # one_hot_next = convert_to_1hot_tensor(next_label, 4)
        # When it is the 9th motion field. Motion field 9 + transform frame 9 = transform frame 10 (pred ES) 
        # Compare it with the ground true ES
        if frame_index == (motion_output.shape[2] - 2):
            one_hot_ES = convert_to_1hot(labels["label_ES"], 4)
            for cate_index, category in enumerate(classes):
                OTS_loss += dice_loss(next_label, one_hot_ES, cate_index) * (1 / weights[category])
        else:
            for cate_index, category in enumerate(classes):
                # loss_forward += dice_loss(one_hot_next, seg_softmax[:, :, frame_index + 1, ...], 
                #                                         cate_index) * (1 / weights[category])
                loss_forward += dice_loss(next_label, seg_softmax[:, :, frame_index + 1, ...], 
                                                         cate_index) * (1 / weights[category])
        flow_source = next_label

    # From ES to frame 2 using backward motion field 9 - 2. Motion field 9 + frame 10 (ES) = transform frame 9 compare with pred segmented 9
    # Backward motion field i + transform frame (i + 1) = transform frame i compare with pred segmented i
    # flow_source = torch.Tensor(labels["label_ES"]).cuda()
    flow_source = convert_to_1hot(labels["label_ES"], 4)
    loss_backward = 0
    
    weights = get_weights(torch.Tensor(labels["label_ES"]), weighted)
    
    # Backward from 9 to 1 
    for frame_index in range(motion_output.shape[2] - 1, 0, -1):
    #for frame_index in range(motion_output.shape[2] - 2, -1, -1):
        backward_motion = motion_output[:, 2:, frame_index,...]
        motion_field = generate_2dmotion_field(flow_source, backward_motion)
        next_label = F.grid_sample(flow_source, motion_field, align_corners=False, mode="bilinear", padding_mode='border')
        
        # When it is the first backward motion field. Backward motion field 1 + transform frame 2 = transform frame 1
        # Compare it with the ground ture ED
        if frame_index == 1:
            one_hot_ED = convert_to_1hot(labels["label_ED"], 4)
            for cate_index, category in enumerate(classes):
                OTS_loss += dice_loss(next_label, one_hot_ED, cate_index) * (1 / weights[category])
        else:
            for cate_index, category in enumerate(classes):
                #loss_backward += dice_loss(one_hot_next, seg_softmax[:, :, frame_index - 1, ...], 
                #                                          cate_index) * (1 / weights[category])
                loss_backward += dice_loss(next_label, seg_softmax[:, :, frame_index - 1, ...], 
                                                          cate_index) * (1 / weights[category])
        flow_source = next_label
        
    return (1 / (2 * (seg_softmax.shape[2] - 2) * omega * len(classes))) * (loss_forward + loss_backward), 1 / (2 * omega * len(classes)) * OTS_loss


def get_weights(labels, weighted):
    """ Get the ratio of all regions of interest from the input labels
        labels shape = N x 1 x H x W
        weighted: If true return a dictionary contained the relative ratio of regions, otherwise a dictionary of one's
    """
    if weighted:
            num_pixels = float(labels.nelement())
            weights = {"Background": torch.sum(labels == 0) / num_pixels,
                       "LV": torch.sum(labels == 1) / num_pixels,
                       "Myo": torch.sum(labels == 2) / num_pixels,
                       "LA": torch.sum(labels == 3) / num_pixels
            }
    else:
        weights = {}
        for category in classes:
            weights[category] = 1

    return weights
