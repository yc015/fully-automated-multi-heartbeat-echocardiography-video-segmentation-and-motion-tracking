from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torch.nn.functional as F
from src.transform_utils import generate_2dmotion_field
from src.utils.echo_utils import makeVideo
from src.utils.camus_validate import labColor_cmap, labColorMap
import cv2
from src.utils.echo_utils import computeSimpsonVolume


# The labColorMap is acquired at https://gitlab.com/stough/camus_segmentation/-/blob/master/src/utils/camus_validate.py#L41
labColorMap = {0: [0, 0, 0], # 0 background
               1: [.3, .3, 0], # 1 LV
               2: [.2, 0, .3], # 2 LV epi, or myocardium
               3: [0, .1, .3]} # 3 LA 


def show_sequence_of_images(sequence_of_images, batch_index=0):
    r""" Show a sequence of image in the LabColorMap
         Sequence of image shape N x D x H x W
    """
    fig, ax = plt.subplots(4, 3, figsize=(9, 15))
    for i in range(4):
        for j in range(3):
            if i * 3 + j < 10:
                image = sequence_of_images[batch_index][i * 3 + j]
                multi_channel = label2LabColorMap(image)
                ax[i][j].imshow(multi_channel)
                ax[i][j].set_title("Frame {: d}".format(i * 3 + j + 1))
            else:
                ax[i][j].axis('off')
    plt.tight_layout()
    
    
def label2LabColorMap(image, alpha=3):
    r""" Expected label shape H x W and on Tensor.cuda or shape H x W on cpu as numpy
         image: the single channel label image to be transferred
         alpha: opacity of the label (or maybe just the intensity)
    """
    try:
        image = image.cpu().detach().numpy()
    # A bold assumption that if the image cannot be detach from the cpu then it is already a numpy array
    except:
        image = image.astype("float64")
    multi_channel = np.zeros(shape=(image.shape[-2], image.shape[-1], 3))
    for channel in range(3):
        multi_channel[..., channel] = image
    for key in labColorMap:
        multi_channel[image==key,:] = alpha * np.array(labColorMap[key])
    
    return multi_channel


def get_deformed_label_forback(label_image, motion_output, grid_mode="nearest"):
    r""" Using the ground truth label ES and ED and forward, backward motion output to get the 
         two series of deformed label from ES to ED and from ED to ES
         label_image: just the batch N x 1 x H x W
         motion_output: the motion output from the motion tracking head N x C x D x H x W
    """
    flow_source_backward = torch.Tensor(label_image["label_ES"])
    backwards = []
    for frame_index in range(9, 0, -1):
        motion_field = generate_2dmotion_field(flow_source_backward, motion_output[:, 2:, frame_index,...])
        new_label = F.grid_sample(flow_source_backward.cuda(), motion_field, align_corners=False, mode=grid_mode, padding_mode='border')
        flow_source_backward = new_label
        backwards.append(new_label)

    flow_source_forward = torch.Tensor(label_image["label_ED"])
    forwards = []
    for frame_index in range(0, 9):
        motion_field = generate_2dmotion_field(flow_source_forward, motion_output[:, :2, frame_index,...])
        new_label = F.grid_sample(flow_source_forward.cuda(), motion_field, align_corners=False, mode=grid_mode, padding_mode='border')
        flow_source_forward = new_label
        forwards.append(new_label)
        
    return forwards, backwards


def get_deformed_image_forback(images, motion_output, grid_mode="bilinear"):
    r""" Prob duplicate of the get_deformed_label_forback. I will wrap two functions together 
         images: input image N x C x D x H x W
         motion_output: motion output from the motion tracking head
         grid_mode: For image deformation, use "bilinear".
    """
    forwards = []
    for frame_index in range(0, 9):
        flow_source_image = images[:, :, frame_index,...]
        motion_field = generate_2dmotion_field(flow_source_image, motion_output[:, :2, frame_index,...])
        new_image = F.grid_sample(flow_source_image.cuda(), motion_field, align_corners=False, mode=grid_mode, padding_mode='border')
        forwards.append(new_image)
        
    backwards = []
    for frame_index in range(9, 0, -1):
        flow_source_image = images[:, :, frame_index,...]
        motion_field = generate_2dmotion_field(flow_source_image, motion_output[:, 2:, frame_index,...])
        new_image = F.grid_sample(flow_source_image.cuda(), motion_field, align_corners=False, mode=grid_mode, padding_mode='border')
        flow_source_image = new_image
        backwards.append(new_image)
    
    return forwards, backwards
    

def apply_sequence_deformation(flow_source_image, motion_output, start_index, end_index, grid_mode="bilinear", forward=True):
    r""" Apply a sequence of deformation to the flow_source_image from the start motion_output to the end
         If forward motion is applied, flow_source_image's frame index == start_index
         If backward motion is applied, flow_source_image's frame index == start_index
         for label deformation grid_mode="nearest" is more prefered
         for image deformation grid_mode="bilinear" works better
         flow_source_image: image or label of one frame shape N x 1 x H x W
         motion_output: motion output from the motion tracking head shape N x 4 x 10 x H x W
         start_index: the start of the motion
         end_index: the end of the motion
         grid_mode: Type of interpolation
         forward: True if forward motion is applied, False backward
    """
    step = 1
    if not forward:
        step = -1
    for frame_index in range(start_index, end_index, step):
        if forward:
            motion_field = generate_2dmotion_field(flow_source_image, motion_output[:, :2, frame_index,...])
        else:
            motion_field = generate_2dmotion_field(flow_source_image, motion_output[:, 2:, frame_index,...])
        new_image = F.grid_sample(flow_source_image, motion_field, align_corners=False, mode=grid_mode, padding_mode='border')
        flow_source_image = new_image
    return new_image


def categorical_dice(truth, pred, k, epi=False, individual=False, epsilon = 1e-7):
    r""" This categorical dice should only be used for getting statistics of the segmented results but not for losses
         truth: The target ED or ES label
         truth shape= N x 1 x H x W
         pred: the predicted segmented output from model after the torch.argmax
         pred shape= N x 1 x H x W
         k: The class index corresponding to the computed dice loss
         The output is a float tensor of the corresponding class dice loss
    """
    assert truth.shape == pred.shape, "Incompatible shape of pred and truth"
    # Dice overlap metric for label value k
    # Matrix holder
    A = torch.Tensor(np.zeros(pred.shape))
    B = torch.Tensor(np.zeros(truth.shape))
    
    if epi:
        # Epi dice is computed by grouping LV and Myo (label 1 and 2)
        A[pred==1] = 1
        A[pred==2] = 1
        B[truth==1] = 1
        B[truth==2] = 1
        
    else:
        A[pred==k] = 1 
        B[truth==k] = 1
    
    if individual:
        return 2 * torch.sum(A * B, dim=(-1, -2)) / (torch.sum(A, dim=(-1, -2)) + torch.sum(B, dim=(-1, -2)) + epsilon)
    else:
        return torch.mean(2 * torch.sum(A * B, dim=(-1, -2)) / (torch.sum(A, dim=(-1, -2)) + torch.sum(B, dim=(-1, -2)) + epsilon))


def get_all_dice(pred, truth, LVepi=False, individual=False):
    r""" Get the dice of all classes = Bg, LV, Myo, LA or Bg, Endo, Epi, LA 
         pred: the pred segmentation output after the torch.argmax shape = N x 1 x H x W
         truth: the ground truth label shape = N x 1 x H x W
         LVepi: True then compute the Bg, Endo, Epi, LA where Endo = LV Epi = LV + Myo
         Dupilcate with the categorical_dice in loss_functions.
         This may be a better implementation.
    """
    dices = {}
    if LVepi:
        for index, category in enumerate(["Background", "Endo", "Epi", "LA"]):
            if category != "Epi":
                dices[category] = categorical_dice(pred, truth, index, individual=individual)
            else:
                dices[category] = categorical_dice(pred, truth, index, epi=True, individual=individual)
        return dices
    else:
        for index, category in enumerate(["Background", "LV", "Myo", "LA"]):
            dices[category] = categorical_dice(pred, truth, index, individual=individual)
        return dices


def show_pred_labels(seg_output_max, label, images, batch_num=0):
    """ display the original image overlay with segmentation, and display the ground true label on side.
        The dice coefficients of key structures are annotated on the overlay image
    """
    fig, ax = plt.subplots(4, 2, figsize=(9, 20))
    for batch_num in range(4):
        lv_dice = categorical_dice(torch.Tensor(label["label_ED"][batch_num][0]).cuda(), seg_output_max[batch_num][0], 1)
        myo_dice = categorical_dice(torch.Tensor(label["label_ED"][batch_num][0]).cuda(), seg_output_max[batch_num][0], 2)
        la_dice = categorical_dice(torch.Tensor(label["label_ED"][batch_num][0]).cuda(), seg_output_max[batch_num][0], 3)
        ax[batch_num][0].imshow(images[batch_num][0][0], cmap="gray")
        ax[batch_num][0].imshow(seg_output_max[batch_num][0].cpu().detach().numpy(), alpha=0.5)
        ax[batch_num][0].set_title("Pred Label ED")
        ax[batch_num][0].text(10, 35, "LV Dice: {:.6f}\nMyo Dice: {:.6f}\nLA Dice: {:.6f}".format(lv_dice, myo_dice, la_dice), c="white")
        ax[batch_num][1].imshow(images[batch_num][0][0], cmap="gray")
        ax[batch_num][1].imshow(label["label_ED"][batch_num][0], alpha=0.5)
        ax[batch_num][1].set_title("Ground True Label ED")
    plt.tight_layout()
    
    
def get_class_pixels(seg_mask):
    """ Get the number of pixels of each classes(Background, LV, Myo, and LA)
        seg_mask: segmentation output after argmax/ground true label shape: N x D(10) x H x W
    """
    bg_pixs = []
    lv_pixs = []
    myo_pixs = []
    la_pixs = []
    for frame_index in range(10):
        bg_pixs.append(float(torch.sum(seg_mask[:, frame_index,...] == 0) // seg_mask.shape[0]))
        lv_pixs.append(float(torch.sum(seg_mask[:, frame_index,...] == 1) // seg_mask.shape[0]))
        myo_pixs.append(float(torch.sum(seg_mask[:, frame_index,...] == 2) // seg_mask.shape[0]))
        la_pixs.append(float(torch.sum(seg_mask[:, frame_index,...] == 3) // seg_mask.shape[0]))
        
    return np.array(bg_pixs), np.array(lv_pixs), np.array(myo_pixs), np.array(la_pixs)


def save_animation_from_images(images_numpy, filename="example_01.gif", fps=4, cmap=None,
                               append_reverse=False):
    """ Using a series of frames (stored in numpy array not tensor) to generate an animation (reflect repeat once)
        and save the animation as an gif
        images_numpy: numpy array of image frames shape = Num_frames x C x H x W
        filename: name of generated gif file
        fps: frame rate (num_frames per second)
    """
    if append_reverse:
        series_img = np.array(np.concatenate((images_numpy, images_numpy[::-1])))
    else:
        series_img = np.array(images_numpy)
    vid_saved = makeVideo(series_img, saved=True, cmap=cmap)
    vid_saved.save(filename, writer='imagemagick', fps=fps,)
    
    
def find_outlier(data):
    # See https://stackoverflow.com/questions/39068214/how-to-count-outliers-for-all-columns-in-python for more reference
    Q1 = np.quantile(data, 0.25)
    Q3 = np.quantile(data, 0.75)
    IQR = Q3 - Q1
    return ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
    
    
def box_plot_outliers(data, ylim=None, title="", ax=None, return_outliers=False):
    if ax:
        ax.boxplot(data)
    else:
        fig, ax = plt.subplots(1, 1)
        ax.boxplot(data)
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
        
    outliers = find_outlier(data)    
    count = outliers.sum()
    total = len(data)
    ax.set_title("{:s}\nMean (+- std): {:.3f} ({:.3f})\nMedian: {:.3f}\nNumber of Outlier: {:d} ({:.2f}%)".format(
        title, np.mean(data), np.std(data), np.median(data), count, (count / total) * 100))
    if return_outliers:
        return outliers
    
def plotContours(img, seg, truth=None, vis=True, save='', alpha = 1.0, ax1=None):
    """ Copied of the plotContours functions in the notebook
    """
    toshow = {'LVepi': [1, 2], 'LVendo':[1], 'LA': [3]}
    order = ['LVepi', 'LA', 'LVendo']
    
    if not ax1:
    # pd = {'alpha': alpha, 'color': tuple([3*x for x in labColorMap[2]])}
        f, (ax0, ax1) = plt.subplots(1,2, figsize=(8,4))
    else:
        ax0 = None
    
    if ax0:
    #     ax[0].set_title('Original')
        ax0.imshow(img, cmap='gray', interpolation=None)

    #     ax[1].set_title('Overlay')
        ax1.imshow(img, cmap='gray', interpolation=None)
    else:
        ax1.imshow(img, cmap='gray', interpolation=None)

    pd = {'alpha':alpha,
          'linewidth':1.5}
    
    for name in order:
        lab = toshow[name]
        
        # Do we have the correct answer.
        if truth is not None:
            truc = truth.copy()
            if len(lab) == 2:
                truc = np.logical_or(truc==lab[0], truc==lab[1]).astype('uint8')
            else:
                truc = (truc == lab[0]).astype('uint8')
            pd['color'] = (.7, .3, .3)
            con_seg, hier_seg = cv2.findContours(truc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Assuming one contour here...
            con_seg = con_seg[0].squeeze()
            con_seg = np.append(con_seg, con_seg[0][None,:], axis=0) # close the loop
            
            ax1.plot(con_seg[:,0], con_seg[:,1], **pd)
        
        
        
        segc = seg.copy()
        if len(lab) == 2:
            segc = np.logical_or(segc==lab[0], segc==lab[1]).astype('uint8')
            pd['color'] = tuple([3*x for x in labColorMap[lab[1]]])
        else:
            segc = (segc == lab[0]).astype('uint8')
            pd['color'] = tuple([3*x for x in labColorMap[lab[0]]])
    
        con_seg, hier_seg = cv2.findContours(segc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Assuming one contour here...
        con_seg = con_seg[0].squeeze()
        con_seg = np.append(con_seg, con_seg[0][None,:], axis=0) # close the loop

        ax1.plot(con_seg[:,0], con_seg[:,1], **pd)
        

    
    # Other ways of doing this result in more empty space, even with tight_layout
    if ax0:
        [a.axes.get_xaxis().set_visible(False) for a in (ax0, ax1)]
        [a.axes.get_yaxis().set_visible(False) for a in (ax0, ax1)]
    else:
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        
    
    plt.tight_layout()
    if len(save) > 0:
        plt.savefig(save, bbox_inches='tight', dpi=300)


# See: https://www.creatis.insa-lyon.fr/Challenge/camus/participation.html
# Modified Color Map for plotting Echonet Segmentation
labColorMap_EchoNet = {0: [0, 0, 0], # 0 background
                       1: [.3, .3, 0]} # 1 LV

# labColor_cmap_EchoNet = ListedColormap(3*np.array([labColorMap[key] for key in range(2)]))

def echonet_overlay(im, lab, lab_gt=None, title='', vis=True, save=False, alpha = 1.0, return_overlay=False):
    '''
    visualization utility to show image and overlayed prospective label.
    If additional argument lab_gt is provided, the vis provides a 
    third axes on which to visualize the difference between the lab and 
    lab_gt images.
    
    im is expected h x w
    lab is expected C x h x w for the network output, or else h x w. 
        - This means lab *can* be the ground truth, or argmax of scores,
          or the probability scores themselves.
    lab_gt is expected to be h x w when provided. I squeeze 3d shapes.
    
    I also hacked it so it returns the overlay image if vis is False. 
    Kinda hacky, cause it doesn't return anything if vis is True. I 
    just didn't want to rewrite most of this code. Should probably 
    modularize better.
    '''
    # Squeeze out the singleton dimensions.
    im = im.copy().squeeze()
    lab = lab.copy().squeeze()

    # Just a three channel version of the echo image.
    if len(im.shape) == 3 and im.shape[2] == 3:
        I = im
    else:
        I = np.stack([im,im,im], axis=-1)
    
    # Special image for lab_gt versus lab.
    if lab_gt is not None:
        if len(lab_gt.shape) == 3: # maybe, for some reason, 1 x h x w was provided...
            lab_gt = lab_gt.squeeze()
            
        if len(im.shape) == 3 and im.shape[2] == 3:
            I_gt = im
        else:
            I_gt = np.stack([im,im,im], axis=-1)
        
        # Make complementary colors for the label/gt difference overlay.
        gtCompColors = {}
        for key, val in labColorMap_EchoNet.items():
            mx = max(val)
            gtCompColors[key] = [mx-x for x in val]

            
    # Determine the kind of lab image we're looking at, original 2d or network output 3d.
    # The network output should be C x h x w
    if len(lab.shape) == 3:
        labSoftMax = softmax(lab, axis=0)
        mchanLabelKey = np.argmax(labSoftMax, axis=0)

        for key in labColorMap_EchoNet:
            whereActive = mchanLabelKey == key

            # This can't be right, making a whole new multichannel temp 
            # and then using only part of it. oh well.
            # prodim is basically an image of a single color, scaled at
            # each pixel by the softmax score.
            prodim = np.multiply(np.stack([labSoftMax[key], 
                                           labSoftMax[key], 
                                           labSoftMax[key]], axis=-1), 
                                 labColorMap_EchoNet[key])

            # We'll only update I where this key/label is the winner.
            I[whereActive,:] += alpha*prodim[whereActive,:]
            
        

    else: # Regular lab image of keys instead of network output. Expected h x w
        # Now add the lab image to the echo.
        mchanLabelKey = lab.copy() # assigning this so the lab_gt stuff still works below
        for key in labColorMap_EchoNet:
            I[lab==key,:] += alpha*np.array(labColorMap_EchoNet[key])
      
    
    if lab_gt is not None:
        for key in [1]: # hardcode, sorry. Want LV to show best.
            whereFP = np.logical_and(mchanLabelKey == key, lab_gt != key)
            whereFN = np.logical_and(mchanLabelKey != key, lab_gt == key)
                
                # Getting the shapes right is difficult.
#                 I_gt[whereFP,:] += np.reshape([0, .2*key, 0], (1,3))
#                 I_gt[whereFN,:] += np.reshape([.2*key, 0, 0], (1,3))
            I_gt[whereFP,:] += labColorMap_EchoNet[key]
            I_gt[whereFN,:] += gtCompColors[key] # false negative is color complement
            
        I_gt = I_gt.clip(0,1)

    I = I.clip(0,1)
    
    if vis is False:
        return I
    
    if return_overlay:
        return I_gt

    
    if lab_gt is not None:
        f, ax = plt.subplots(1,3, figsize=(9,4))
    else:
        f, ax = plt.subplots(1,2, figsize=(6,4))
        
    # Regardless of lab_gt the first and second axes don't change.    
    ax[0].imshow(im, cmap='gray', interpolation=None)
    ax[1].imshow(I, interpolation=None)
    
    
    if lab_gt is not None:
        ax[2].imshow(I_gt, interpolation=None)
    
    
    # Other ways of doing this result in more empty space, even with tight_layout
    [a.axes.get_xaxis().set_visible(False) for a in ax]
    [a.axes.get_yaxis().set_visible(False) for a in ax]

    if len(title) > 0:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()
    
    
    if save:
        # Save the plot, use title info
        savename = title.replace(' ', '_').replace(',', '').lower() + '_overlay'
        plt.savefig(savename+'.pdf', bbox_inches='tight')
        plt.savefig(savename+'.png', dpi=200, transparent=True, bbox_inches='tight')
    
    return ax[0]

def make_annotated_gif(segmentations, video, filename="example.gif"):
    gray_video = np.dot(video.transpose([1, 2, 3, 0]), np.array([.2989, .5870, .1140]))
    
    overlay_video = []

    for index in range(gray_video.shape[0]):
        overlay_frame = (echonet_overlay(gray_video[index], segmentations[index], vis=False))
        overlay_video.append(overlay_frame)

    overlay_video = np.array(overlay_video)
    
    lv_volume = []
    for i in range(len(segmentations)):
        output_seg = segmentations[i]
        volume = computeSimpsonVolume(output_seg, output_seg, (1.0, 1.0), (1.0, 1.0))
        lv_volume.append(volume)
    
    lv_volume = np.array(lv_volume)

    fig = Figure(figsize=(7, 0.4), facecolor=(0., 0., 0., 1.0))
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ax.text(0.27,0.0, "LV Volume (ml)", c='limegreen', fontsize=26)
    ax.axis('off')
    plt.tight_layout()

    canvas.draw()       # draw the canvas, cache the renderer

    width, height = fig.get_size_inches() * fig.get_dpi()
    title = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    
    images = []
    for i in range(len(lv_volume)):
        fig = Figure(figsize=(7, 0.8), facecolor=(0., 0., 0., 1.0))
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.set_facecolor((0, 0, 0, 1.0))
        ax.set_ylim(min(lv_volume) - 100, max(lv_volume) + 100)
        xmin, xmax = ax.set_xlim(0, len(lv_volume) + 1)
        ax.plot(np.arange(i+1), lv_volume[:i+1], c='limegreen')
        ax.axis('off')
        plt.tight_layout()

        canvas.draw()       # draw the canvas, cache the renderer

        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        images.append(img)
        
    for i in range(len(images)):
        images[i] = np.concatenate([title.copy(), images[i]])[:, 77: 630]
            
    overlay_resized_video = []
    images = np.array(images).astype("float64")
    images /= images.max()

    for i in range(len(overlay_video)):
        resized_image = cv2.resize(overlay_video[i], dsize=(images[0].shape[1], images[0].shape[1]), interpolation=cv2.INTER_NEAREST)
        overlay_resized_video.append(np.concatenate([resized_image, images[i]]))
        
    save_animation_from_images(overlay_resized_video,
                               filename=filename, 
                               fps=30)