'''
CAMUS dataset visualization and validation utilities.
Joshua Stough

Basically abstracting out some of the repeated tedium in CAMUS notebook 
experiments.

See the fully-encapsulated CAMUS_UNet.ipynb
'''

'''
This function shows a simple 1x2 plot, with the grayscale echo in the left, and 
an overlay of the lab(el) image on the right.

It is supposed to work both if the label image is the original (1x) h x w, or 
also if it's the network output (4 x h x w in the case of four classes, which 
I've hardcoded here).

Comparing lab == key won't work if the lab image is the network output. 
In that case argmax in the channels dimension would tell us what label is best
for that pixel. 

But we should also weight the color by the softmax of lab...
That was quite a bit to get working (see prodim below), I must be thinking
about it wrong. 
'''

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap # , LinearSegmentedColormap
import numpy as np

from scipy.special import softmax

# import cv2 # for morphology, maybe their connected component stuff is faster...
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes



# See: https://www.creatis.insa-lyon.fr/Challenge/camus/participation.html
labColorMap = {0: [0, 0, 0], # 0 background
               1: [.3, .3, 0], # 1 LV
               2: [.2, 0, .3], # 2 LV epi, or myocardium
               3: [0, .1, .3]} # 3 LA

labColor_cmap = ListedColormap(3*np.array([labColorMap[key] for key in range(4)]))

def camus_overlay(im, lab, lab_gt=None, title='', vis=True, save=False, alpha = 1.0, return_overlay=False):
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
    im = im.squeeze()
    lab = lab.squeeze()

    # Just a three channel version of the echo image.
    I = np.stack([im,im,im], axis=-1)
    
    # Special image for lab_gt versus lab.
    if lab_gt is not None:
        if len(lab_gt.shape) == 3: # maybe, for some reason, 1 x h x w was provided...
            lab_gt = lab_gt.squeeze()
            
        I_gt = np.stack([im,im,im], axis=-1)
        
        # Make complementary colors for the label/gt difference overlay.
        gtCompColors = {}
        for key, val in labColorMap.items():
            mx = max(val)
            gtCompColors[key] = [mx-x for x in val]

            
    # Determine the kind of lab image we're looking at, original 2d or network output 3d.
    # The network output should be C x h x w
    if len(lab.shape) == 3:
        labSoftMax = softmax(lab, axis=0)
        mchanLabelKey = np.argmax(labSoftMax, axis=0)

        for key in labColorMap:
            whereActive = mchanLabelKey == key

            # This can't be right, making a whole new multichannel temp 
            # and then using only part of it. oh well.
            # prodim is basically an image of a single color, scaled at
            # each pixel by the softmax score.
            prodim = np.multiply(np.stack([labSoftMax[key], 
                                           labSoftMax[key], 
                                           labSoftMax[key]], axis=-1), 
                                 labColorMap[key])

            # We'll only update I where this key/label is the winner.
            I[whereActive,:] += alpha*prodim[whereActive,:]
            
        

    else: # Regular lab image of keys instead of network output. Expected h x w
        # Now add the lab image to the echo.
        mchanLabelKey = lab.copy() # assigning this so the lab_gt stuff still works below
        for key in labColorMap:
            I[lab==key,:] += alpha*np.array(labColorMap[key])
      
    
    if lab_gt is not None:
        for key in [3,2,1]: # hardcode, sorry. Want LV to show best.
            whereFP = np.logical_and(mchanLabelKey == key, lab_gt != key)
            whereFN = np.logical_and(mchanLabelKey != key, lab_gt == key)
                
                # Getting the shapes right is difficult.
#                 I_gt[whereFP,:] += np.reshape([0, .2*key, 0], (1,3))
#                 I_gt[whereFN,:] += np.reshape([.2*key, 0, 0], (1,3))
            I_gt[whereFP,:] += labColorMap[key]
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
    
    
    
# Computing Dice for the labels.
labNameMap = {0: 'Background',
              1: 'Left Ventricle',
              2: 'Myocardium',
              3: 'Left Atrium'}

nameLabMap = dict([(labNameMap[key], key) for key in labNameMap]) 

'''
getDices(autoseg, labels): given (N, C, H, W) autoseg batch and 
(N, 1, H, W) labels ground truth (N is batch size, C is 4 or the
number of keys in above labNameMap), return (N, C) Dice scores.
'''
def getDices(autoseg, labels):
    
    # Similar to code in visOverlay, where we want to find the choice made by
    # the autoseg
    labSoftMax = softmax(autoseg, axis=1)
    mchanLabelKey = np.argmax(labSoftMax, axis=1) # should be (N, H, W)
    
    labels = labels.squeeze() # should be (N, H, W) 
    
    retDice = np.zeros((autoseg.shape[0], len(labNameMap)))
    
    for case in range(autoseg.shape[0]):
        for key in labNameMap:
            seg = (mchanLabelKey[case]==key).astype(np.uint8)
            lab = (labels[case]==key).astype(np.uint8)
            
#             print(seg.max(), lab.max()) debug
            # Dice is intersection over average
#             print('lab {}, seg {}, intersect {}'.format(lab.sum(), seg.sum(), (seg*lab).sum()))
            retDice[case][key] = (2.0*((seg*lab).sum()))/(seg.sum() + lab.sum())
    
    return retDice

# Little dictionary extend, where the nparr in (key, nparr) is extended.
#             d[key].extend(d_add[key]) # For lists. 
# Also adds keys that aren't already in the d.
def dict_extend_values(d, d_add):
    new_d = {}
    for key in d:
        if key in d_add:
            new_d[key] = np.concatenate([d[key], d_add[key]])
        else:
            new_d[key] = d[key]
            
    for key in d_add:
        if key not in d:
            new_d[key] = d_add[key]
    return new_d

'''
getDiceByName(autoseg, labels, combs={'LVepi': [1, 2]}): 
Rather than return a (N,C) as above, and let the caller deal with it,
this function returns a dictionary of (name, list) where the name 
is label/object of interest/combination of objects, and list is length 
N dice results over the batch.

Given (N, C, H, W) autoseg batch or cleaned mchanLabelKey
and 
(N, 1, H, W) labels ground truth.
'''
def camus_dice_by_name(autoseg, labels, combs={'LVepi': [1, 2]}):
    if len(autoseg.shape) == 4: 
        labSoftMax = softmax(autoseg, axis=1)
        mchanLabelKey = np.argmax(labSoftMax, axis=1) # should be (N, H, W)
    elif len(autoseg.shape) == 3:
        mchanLabelKey = autoseg.copy()
    
    if len(labels.shape) == 4: # N x (C or 1) x H x W
        if labels.shape[1] > 1: # labels are one-hotted, re-shrink them.
            labels = np.argmax(labels, axis=1)
        else:
            labels = np.squeeze(labels, axis=1) 
    # caused a bug if N  =1 and there is  no C. 
    # else:
    #     labels = labels.squeeze() 
    # labels should be (N, H, W) 
    
    retDice = dict([(name, []) for name in nameLabMap] + \
                   [(name, []) for name in combs])
    
    
    for case in range(autoseg.shape[0]):
        for key in nameLabMap:
            seg = (mchanLabelKey[case]==nameLabMap[key]).astype(np.uint8)
            lab = (labels[case]==nameLabMap[key]).astype(np.uint8)
            
            retDice[key].append((2.0*((seg*lab).sum()))/(seg.sum() + lab.sum()))
            
        for key in combs:
            segSoFar, labSoFar = np.zeros(mchanLabelKey.shape[1:]), np.zeros(labels.shape[1:])
            for index in combs[key]:
                segSoFar += (mchanLabelKey[case]==index).astype(np.uint8)
                labSoFar += (labels[case]==index).astype(np.uint8)
            
            assert segSoFar.max() < 2 and labSoFar.max() < 2,\
                'camus_dice_by_name ERROR: combination labels '\
                'should not overlap, got max {} and {}.'.format(segSoFar.max(),
                                                                labSoFar.max())
            
            retDice[key].append((2.0*((segSoFar*labSoFar).sum()))/ \
                                (segSoFar.sum() + labSoFar.sum()))
    
    return retDice


# I'm putting this stuff here because it relies on the labNameMap 
# This is one step
def cleanupBinary(abin, holesize=128, connectivity=1):
    '''
    cleanupBinary(abin, holesize=100, connectivity=1): keep only largest connected component and
    fill holes on binary image.
    '''
    cc = label(abin)
    
    # get the largest region when filled.
    regions = regionprops(cc)
    if len(regions) == 0:
        return None
    
    largestR = regions[np.argsort([x.filled_area for x in regions])[-1]]
    
    largfilled = remove_small_holes((cc == largestR.label), 
                                    area_threshold=holesize, 
                                    connectivity=connectivity)
    return largfilled.astype('int')
    

# This is for the whole output
def cleanupSegmentation(seg, holesize=128, connectivity=1):
    '''
    cleanupSegmentation(seg, holesize=128, connectivity=1): return the segmentation
    where for each label we have kept only the largest connected component and filled
    its holes. Expects [frames x ] [scores x] h x w
    '''
    seg_shape = seg.shape
    
    assert len(seg_shape) >= 2 and len(seg_shape) < 5,\
        'cleanupSegmentation: requires input shape len [2,4], not {}'.format(seg_shape)
    
    if len(seg_shape) == 4: # A whole video or batch of scores, argmax it to frames x h x w
        labSoftMax = softmax(seg, axis=1)
        seg = np.argmax(labSoftMax, axis=1) 
    
    elif len(seg_shape) == 3 and seg_shape[0] == len(labNameMap): 
        # special case, channels x h x w. single frame of the scores.
        labSoftMax = softmax(seg, axis=0)
        seg = np.expand_dims(np.argmax(labSoftMax, axis=0), axis=0)
        
    elif len(seg_shape) == 2: # Was just a single frame of label numbers. Add a dim for the lines below
        seg = np.expand_dims(seg, axis=0)

    
    
    # Now we know it's three dimensional, batch/frames x h x w
    # Rewrite in numpy
    
#     cleaned = []
    cleanedSoFar = np.array([])
    
    for frame in seg:
        cI = np.zeros_like(frame)
        for lab in labNameMap.keys():
            part = cleanupBinary((frame==lab), holesize=holesize)
#             if np.any(cI[part]): # should not happen...and it's really slow to check...
#                 print('overwriting non-zero label.')
            if np.any(part):
                # Due to the boundary vagaries of skimage, this could end up adding to a 
                # pixel more than once. Need a fancier check.
                # cI = cI + lab*part
                cI = np.where(part, lab*part, cI)
        # the list/stack approach prepends a new dimension.
        # With np approach, do it ourselves.
        cI = np.expand_dims(cI, axis=0)
#         cleaned.append(cI)
        # Now cat this part with previous parts
        if len(cleanedSoFar) == 0:
            cleanedSoFar = cI.copy()
        else:
            cleanedSoFar = np.concatenate([cleanedSoFar, cI], axis=0)
    
#     return np.stack(cleaned).squeeze() # squeeze in case single frame. 
    return cleanedSoFar.squeeze()
    