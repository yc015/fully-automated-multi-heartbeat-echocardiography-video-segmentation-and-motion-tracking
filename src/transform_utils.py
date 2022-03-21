from skimage.transform import rescale, resize
import os
import numpy as np
import SimpleITK as itk
from scipy.interpolate import interp1d
import torch
from torch import nn
from src.utils.camus_transforms import random_GaussNoiser
import cv2
import torch.nn.functional as F
from skimage.transform import rotate


def generate_2dmotion_field(x, offset):
    # Qin's code for joint_motion_seg learning works fine on our purpose too
    # Same idea https://discuss.pytorch.org/t/warp-video-frame-from-optical-flow/6013/5
    x_shape = x.size()
    grid_w, grid_h = torch.meshgrid([torch.linspace(-1, 1, x_shape[2]), torch.linspace(-1, 1, x_shape[3])])  # (h, w)
    grid_w = grid_w.cuda().float()
    grid_h = grid_h.cuda().float()

    grid_w = nn.Parameter(grid_w, requires_grad=False)
    grid_h = nn.Parameter(grid_h, requires_grad=False)

    offset_h, offset_w = torch.split(offset, 1, 1)
    offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
    offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)

    offset_w = grid_w + offset_w
    offset_h = grid_h + offset_h
    
    offsets = torch.stack((offset_h, offset_w), 3)

    return offsets


# The classes in the echo_utils.camsu_transforms are rewritten to accommodate with usage of the sequences of images with/without labels insteand of pairs of image with ground true label. 

# Load and resample image sequence
class LoadImageSequenceToNumpy(object):
    """ Load the a sequence of echo video frames 
        The number of frames sampled out from the sequence is 10
        which is the minimum number of frames available for the video
        in CAMUS data set. The interval between each sampled frame is same.
        
        The order of the sampled frames is always from ED to ES inclusively. """
    # Rewritten and combine the LoadSITKFromFilename and SitkToNumpy
    def __init__(self, field, normed=True, negative_normed=False, num_frames=10):
        self.num_frames = num_frames
        self.field = field
        self.normed = normed
        self.negative_normed = negative_normed
        
    def __call__(self, data):
        entries = []
        
        for entry in data[self.field]:
            is_images = False
            rev_order = False
            # Search the tag that indicate the order of the image sequences
            if entry.find("ES-ED") >= 0:
                # Found tag ES-ED, then the entry is a sequence of cardiac images from ES to ED, reverse the sequence order when loading
                is_images = True
                rev_order = True
            elif entry.find("ED-ES") >= 0:
                # Found tag ED-ES, then the entry is a sequence of cardiac images from ED to ES, preserve the sequence order
                is_images = True
                rev_order = False
            # If no tag is found, by default, the entry is a label without order
            if is_images:
                itk_imgs = itk.ReadImage(entry[:len(entry)-5])
                temp = itk.GetArrayFromImage(itk_imgs).astype(np.float64)
                if rev_order:
                    # We want to ensure the order is always from ED to ES
                    temp = temp[::-1]
                if self.normed:
                    # If normalization is required
                    #for i in range(temp.shape[0]):
                        # If [-1, 1] range normalization is required
                        # Actually this part can be change to cv2.normalized(img, None, alpha=lower_bound, beta=upper_bound,...)
                        # And discard the self.negative_normed 
                        #temp[i] = cv2.normalize(temp[i], None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX)
                    if self.negative_normed:
                        temp -= temp.min()
                        temp /= temp.max()
                        temp -= 0.5
                        temp *= 2
#                         temp = cv2.normalize(temp.transpose([1, 2, 0]), None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX)
#                         temp = temp.transpose([2, 0, 1])
                    else:
                        #temp[i] = (temp[i] - np.min(temp[i])) / (np.max(temp[i]) - np.min(temp[i]))
#                         temp = cv2.normalize(temp.transpose([1, 2, 0]), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#                         temp = temp.transpose([2, 0, 1])
                        temp -= temp.min()
                        temp /= temp.max()
                            
                if len(temp) >= self.num_frames:
                    # Resample the sequence of images to 10 frames
                    # sample_index = get_resample_index(len(temp), 10)
                    # temp = temp[sample_index]
                    temp = torch.Tensor(temp.copy()).unsqueeze(0).unsqueeze(0)
                    temp = F.interpolate(temp, size=(self.num_frames, temp.shape[-2], temp.shape[-1]), mode="trilinear", align_corners=False)
                    temp = temp.numpy()[0, 0,...]
            else:
                itk_label = itk.ReadImage(entry)
                temp = itk.GetArrayFromImage(itk_label).astype(np.long)
            
            entries.append(temp)
        
        # Now modify the dictionary that was sent in.
        data[self.field] = entries
        return data


def get_resample_index(num_frames, sample_amount=10):
    """ Not used anymore! """
    """ Get the resample indexes based on the number of frames in sequences and the amount of samples we want to extract """
    index_array = np.arange(num_frames)
    possible_index = np.arange(0, num_frames, (num_frames-1) / (sample_amount-1))
    
    # Interpolate the index array to get the 10 sample indexes with equal intervals
    nearest = interp1d(np.arange(len(index_array)), index_array, kind='nearest')
    nearest_int_index = nearest(possible_index)
    nearest_int_index[-1] = num_frames - 1
    
    return nearest_int_index.astype('int16')


class ResizeImage(object):
    r""" Rewritten code of the ResizeImagesAndLabels in camus_transforms
         Works both for image sequences or single image or single label
    """
    def __init__(self, size, image_field, is_sequence=False, is_label=False, anti_aliasing=True, order=1):
        self.size = size
        self.is_sequence = is_sequence
        self.is_label = is_label
        self.image_field = image_field
        self.anti_aliasing = anti_aliasing
        self.order = order
        
    def __call__(self, data):
        image_entries = []

        image_field = self.image_field
        
        for image_entry in data[image_field]:
            # We can assert here that each pairs at least starts as the same size.
            
            # Resize the image and label.
            # https://scikit-image.org/docs/dev/api/skimage.transform.html#resize
            
#            print(image_entry.dtype) input was float32
            if self.is_sequence:
                image_entry = np.expand_dims(image_entry, 0)
            if self.is_sequence:
                modified_entry = []
                for i in range(len(image_entry)):
                    # transpose to go from chan x h x w to h x w x chan and back.
                    modified_image = resize(image_entry[i].transpose([1,2,0]), 
                                         self.size, mode='constant', 
                                         anti_aliasing=self.anti_aliasing,
                                         order=self.order,
                                         preserve_range=True)
                    modified_image = modified_image.transpose([2,0,1])

                    modified_entry.append(modified_image)
                    # at this point they were float64, not good.  
                    # I guess skimage used the native float type.

                if self.is_label:
                    image_entries.append(modified_entry.astype(np.long))
                else:
                    image_entries.append(np.array(modified_entry).astype(np.float32))
            
            else:
                image_entry = resize(image_entry.transpose([1,2,0]), 
                                     self.size, mode='constant', 
                                     anti_aliasing=self.anti_aliasing,
                                     order = self.order, 
                                     preserve_range=True)
                image_entry = image_entry.transpose([2,0,1])
                if self.is_label:
                    image_entries.append(image_entry.astype(np.long))
                else:
                    image_entries.append(np.array(image_entry).astype(np.float32))
        
        # Now, after resizing all the image/label pairs, modify the dictionary accordingly.
        data[self.image_field] = image_entries
        
        return data

    
class random_video_windower(object):
    '''
    random_Windower: Used to randomly window contents of a matrix (image, video). 
    e.g., [0.5, 1.0] means pick a random subrange in [.5, 1] of the input 
        range and scale it to the entire input range, then clip/saturate 
        the remaining complement of that subrange.
    '''
    def __init__(self, scale):
        scale = sorted(scale)
        assert min(scale) > 0.0 and max(scale) <= 1.0 and min(scale) <= max(scale),\
            'random_Windower: scale range {} must be in (0,1].'.format(scale)
        
        self.scale = scale
    
    def __call__(self, images):
        images = images.transpose(1, 2, 0)
        imgs_min, imgs_max = images.min((0, 1)), images.max((0, 1))
        
        # Now scale represents a list of high and low random scale to use.
        # Get the random scale to use in this case.
        sc_ = self.scale[0] + (self.scale[1]-self.scale[0])*np.random.rand()
        # Now basically the same as above.
        imgs_range = imgs_max-imgs_min
        locut = imgs_min + imgs_range*((1.0-sc_)*np.random.rand()) # where to start the window.
        hicut = locut+(sc_*imgs_range)
            
        # We're mapping [locut,hicut] to [im_min,im_max] and clipping outside that range.
        alpha = (images-locut)/(hicut-locut)
        images = (1.0-alpha)*imgs_min + alpha*imgs_max
        for i in range(images.shape[-1]):
            images[...,i] = np.clip(images[...,i], imgs_min[i], imgs_max[i])
        
        return images.transpose(2, 0, 1)
    
    
class WindowImage(object):
    ''' Rewritten code of the WindowImagesAndLabels in camus_transforms.py
    '''
    
    def __init__(self, scale=[1.0, 1.0], image_field='image_sequence', is_sequence=True, is_label=False):
        # only if scale is scalar.
#         assert scale > 0.0 and scale <= 1.0,\
#             'WindowImagesAndLabels: scale {} must be in (0,1].'.format(scale)

#         scale = sorted(scale)
#         assert min(scale) > 0.0 and max(scale) <= 1.0 and min(scale) <= max(scale),\
#             'random_Windower: scale range {} must be in (0,1].'.format(scale)
        
#         self.scale = scale
        
        self.image_field = image_field
        self.is_sequence = is_sequence
        self.is_label = is_label
        self.windower = random_video_windower(scale)
        
    def __call__(self, data):
        image_entries = []
        image_field = self.image_field
        
        if self.is_sequence:
            for image_sequence in data[image_field]:
                windowed_images = self.windower(np.squeeze(image_sequence.astype(np.float32)))
                image_entries.append(np.expand_dims(windowed_images, 0))
        else:
            for image_entry in data[image_field]:
                image_entry = self.windower(image_entry)
                if self.is_label:
                    # should not window the label
                    image_entries.append(image_entry.astype(np.long))
                else:
                    image_entries.append(image_entry.astype(np.float32))
                    
        data[self.image_field] = image_entries

        return data
    

class random_Rotater(object):
    def __init__(self, scalestd, rtype, order):
        assert scalestd >= 0.0 and scalestd <= 60.0,\
            'RotateImagesAndLabels: scale {} must be in [0,60].'.format(scale)
        
        assert rtype in ['normal', 'uniform'],\
            'RotateImagesAndLabels: rtype must be of ["normal", "uniform"]'
        
        self.scalestd = scalestd
        self.rtype = rtype
        self.order = order
        
    def __call__(self, img, rot_deg=None):
        if not rot_deg:
            # Random angle to rotate by.
            # Either +- scale if uniform or +- 3scale if normal.
            if self.rtype == 'normal':
                rot_deg = self.scale*np.random.randn()
                rot_deg = np.clip(rot_deg, -3*self.scale, 3*self.scale)
            else: 
                rot_deg = 2.0*self.scale*np.random.rand() - self.scale
        
        # the rotational center, in col,row format.
        # Assuming h x w or 1 x h x w, w/2 is the x coordinate of the top middle.
        center = (img.shape[-1]/2.0 - 0.5, 0.5)
            
        # skimage expects channels last
        # transpose to go from chan x h x w to h x w x chan and back. labels is 2d.
        
        if len(img.shape) == 3:
            img = rotate(img.transpose([1,2,0]),
                         angle=rot_deg,
                         center=center,
                         order = self.order, # order 0 for nearest neighbor, 1 spline
                         preserve_range=True
                        )
            img = img.transpose([2,0,1])
        else:
            img = rotate(img, # already 2d.
                         angle=rot_deg,
                         center=center,
                         order=self.order, # order 0 for nearest neighbor,
                         preserve_range=True
                        )
            
        # If order 0, then we assume long type outputs.
        if self.order:
            img = img.astype(np.float32)
        else:
            img = img.astype(np.long)

        return img
    
    
class RotateVideoAndLabels(object):
    '''
    RotateImages: Rotates same as above, but allows for just one to be rotated. 
    Like in the autoencoding case, when we wouldn't need to manipulate both the 
    label and label again.
    Not true: the input to the network may be noised, so it's distinct from the
    expected output.
    '''
    def __init__(self, 
                 scalestd=0.0, 
                 rtype='normal',
                 image_field='image_sequence',
                 ed_field="label_ED",
                 es_field="label_ES",
                 image_order=1,
                 label_order=0
                ):
        # only if scale is scalar.
#         assert scale > 0.0 and scale <= 1.0,\
#             'WindowImagesAndLabels: scale {} must be in (0,1].'.format(scale)
        assert scalestd >= 0.0 and scalestd <= 60.0,\
            'RotateImagesAndLabels: scale {} must be in [0,60].'.format(scale)
        
        assert rtype in ['normal', 'uniform'],\
            'RotateImagesAndLabels: rtype must be of ["normal", "uniform"]'
        
        self.scale = scalestd
        self.rtype = rtype
        self.image_field = image_field
        self.ed_field = ed_field
        self.es_field = es_field
        self.image_rotater = random_Rotater(scalestd, rtype, image_order)
        self.label_rotater = random_Rotater(scalestd, rtype, label_order)
        
    def __call__(self, data):
        image_entries = []
        ed_entries = []
        es_entries = []

        
        for image_entry, ed_entry, es_entry in zip(data[self.image_field], 
                                            data[self.ed_field], data[self.es_field]):
            # Random angle to rotate by.
            # Either +- scale if uniform or +- 3scale if normal.
            if self.rtype == 'normal':
                rot_deg = self.scale*np.random.randn()
                rot_deg = np.clip(rot_deg, -3*self.scale, 3*self.scale)
            else: 
                rot_deg = 2.0*self.scale*np.random.rand() - self.scale

            
            # 3 sigma should mean 3 out of 1000 end up being clipped, 
            # hopefully that's not too bad.
            # print('rotating by {} '.format(rot_deg))
            rotated_images = self.image_rotater(np.squeeze(image_entry), rot_deg)
            image_entries.append(np.expand_dims(rotated_images, 0))
            ed_entries.append(self.label_rotater(ed_entry, rot_deg))
            es_entries.append(self.label_rotater(es_entry, rot_deg))
        
        # Now, after resizing all the image/label pairs, modify the dictionary accordingly.
        data[self.image_field] = image_entries
        data[self.ed_field] = ed_entries
        data[self.es_field] = es_entries

        return data 
    
    
class GaussianNoiseImageSequence(object):
    ''' Rewritten code of the GaussianNoiseEcho in camus_transforms.py
        Now, it works for a sequence of images
    '''
    def __init__(self, 
                 sig_range=[0.0, 0.0],
                 image_field='image_sequence'):
        
        self.image_field = image_field
        self.noiser = random_GaussNoiser(sig_range)
        
    def __call__(self, data):
        entries = []
        
        for entry in data[self.image_field]:
            modified_entry = []
            entry = np.squeeze(entry)
            for image in entry:
                modified_entry.append(self.noiser(image))
            entries.append(np.expand_dims(np.array(modified_entry),0))
        
        data[self.image_field] = entries
        
        return data
    

class NormalizedImageSequence(object):
    ''' Helper Transformation that use the cv2.normalize to renormalize an image/images into a defined range
    '''
    def __init__(self, 
                 normalized_range=[0.0, 1.0],
                 norm_type=cv2.NORM_MINMAX,
                 image_field='image_sequence'):
        
        self.normalized_range = normalized_range
        self.norm_type = norm_type
        self.image_field = image_field
        
    def __call__(self, data):
        entries = []
        
        for entry in data[self.image_field]:
            # From 1 x D(10) x H x W to D(10) x H x W
            entry = np.squeeze(entry)
            entry = cv2.normalize(entry, None, alpha=self.normalized_range[0], beta=self.normalized_range[1], norm_type=self.norm_type)
            entry = np.expand_dims(entry, 0)
            entries.append(entry)
        
        data[self.image_field] = entries
        
        return data


def make_camus_echo_dataset(rootpath="CAMUS_data/training/", view = '2CH', drop_ES_ED=False):
    # Rewritten code of make_camus_echo_dataset in echo_utils.camus_load_info
    dataByPat = {}
    
#     for root, subdirs, files in os.walk(rootpath):
#         templist = [os.path.join(root, filename) for filename in files if 'mhd' in filename]
#         dataset.extend([f for f in templist if chamber in f and 'seq' not in f])

    patients = os.listdir(rootpath)
    info_filename = 'Info_{}.cfg'.format(view)
    
    for pat in patients:
        subdir = os.path.join(rootpath, pat)
        files = os.listdir(subdir)
        if len(files) > 0:
            # Use the electronic health record data to determine the order of the video frames
            with open(os.path.join(subdir, info_filename), 'r') as thefile:
                datums = thefile.read().split('\n')
                data_dict = dict([dat.split(': ') for dat in datums[:-1]])
                ED_fr = int(data_dict['ED']) # Frame index corresponding to the ED frame
                ES_fr = int(data_dict['ES']) # Frame index corresponding to the ES frame
                if ED_fr < ES_fr:
                    order = "ED-ES"
                elif drop_ES_ED:
                    continue
                else:
                    order = "ES-ED"
                # Append the order at the end of image sequence file name
                # Not a good and safe way to record the order information. I will change this.

            dataset = []
            dataset.append({
                'pat_view_phase': [(pat, view, 'sequence')],
                'image_sequence': [os.path.join(subdir, '{}_{}_{}.mhd{}'.format(pat, view, 'sequence', order))],
                'image_ED': [os.path.join(subdir, '{}_{}_{}.mhd'.format(pat, view, 'ED'))],
                'image_ES': [os.path.join(subdir, '{}_{}_{}.mhd'.format(pat, view, 'ES'))],
                'label_ED': [os.path.join(subdir, '{}_{}_{}_gt.mhd'.format(pat, view, 'ED'))],
                'label_ES': [os.path.join(subdir, '{}_{}_{}_gt.mhd'.format(pat, view, 'ES'))]
            })
            dataByPat[pat] = list(dataset)
            
        
    return dataByPat
