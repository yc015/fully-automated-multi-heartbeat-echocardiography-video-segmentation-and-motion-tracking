'''
Transformations of the CAMUS dataset.
Joshua Stough

https://www.creatis.insa-lyon.fr/Challenge/camus/

Load and transform the CAMUS dataset for pytorch.
See CAMUS_UNet.ipynb for a full implementation 
and discussion.

These transforms were written to work with the list of dictionaries format 
often used in pytorch, where each entry starts as a filename and gets
iteratively processed by a sequence of transforms to be the input or output
to a network. 
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

A lot of the code is duplicated in echo_utils, which is dangerous. For the 
non-random (normalize, resize) and random operations (rotation, windowing), 
there should be common code called by all these. The code here just has to 
manipulate the list of dictionaries, compared to the echo_utils code that 
should work straight on numpy arrays.

To try to keep code duplication down, I'll try to write simpler random_Windower
and random_Rotater and other classes. Then each transformer pytorch uses simply
instantiates one those to do the actual numpy manipulation...

SimpleITK: http://www.simpleitk.org/SimpleITK/help/documentation.html
    and sample notebooks: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/
'''

import numpy as np
import SimpleITK as itk
from skimage.transform import (rescale, 
                               resize, 
                               rotate)


'''
This first one basically takes a list of filenames (the contents of data[self.field]) and 
replaces it with a list of loaded SITK images. The way it's called first though, I think 
the input list (the value associated with the field key of the input argument data) contains
only one filename. But anyway...

See camus_load_info.py on generating the input list for this to run on.
'''
class LoadSITKFromFilename(object):
    def __init__(self, field):
        self.field = field
        
    def __call__(self, data):
        entries = []
        
#         print('LoadSITKFromFilename: Attempting to read data {}'.format(data), end='')
        
        for entry in data[self.field]:
            entries.append(itk.ReadImage(entry))
        
        # Now modify the dictionary that was sent in.
        data[self.field] = entries
        return data
    
    
    
'''
Second transform is to convert now the list of SITK image objects into a list of numpy arrays 
(one for each transformed image). We just get the image, turn it into a float numpy array, and 
scale it to [0,1].

Remember, sitk and numpy have [reversed order dimensions]
(http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/
01_Image_Basics.html#The-order-of-index-and-dimensions-need-careful-attention-during-conversion), 
so x,y,z becomes z,y,x

From CAMUS_play, we know the SITK images are w x h x 1, so the numpy that comes out is
1 x h x w, and we'll keep it channels first.

Need a hack here: The label images should not be normed. To avoid that I could
either look at the field argument itself, maybe not the best, or maybe add a
normed parameter to the init? Yes, the second worked.

'''
class SitkToNumpy(object):
    def __init__(self, field, normed=True):
        self.field = field
        self.normed = normed
        
    def __call__(self, data):
        entries = []
        
        for entry in data[self.field]:
            
            if self.normed:
                temp = itk.GetArrayFromImage(entry).astype(np.float32)
            
                # Norm to [0,1]. Separtely normalize each channel if there's more than one. 
                # Alternatively, this will norm each frame of video separately.
                # Also, the ellipses to specify later dimensions are unnecessary.
            
                for channel in range(temp.shape[0]):
                    temp[channel] = (temp[channel] - np.min(temp[channel])) / \
                                    (np.max(temp[channel]) - np.min(temp[channel]))
            else:
                # Assuming it's the label type, needs to be long, not float.
                temp = itk.GetArrayFromImage(entry).astype(np.long)
            
            entries.append(temp)
        
        data[self.field] = entries
        
        return data

    
    
'''
Third transform is to resize the image to the specified size, and that's it.
Easier than the VNET example, where they do some center crop. This part may get more
complicated as I iterate. This corresponds to CropCenteredSubVolume in the VNET example.

Here we're expecting channels-first images, that we'll resize simply through 
skimage.transforms.resize

the label has only one channel, but the image could have more than one. that's
why the shape checking starts at 1 (to ignore the channels). This part would need more 
thought for video, where we might only have the label at two time points. That means
the number of frames and the number of channels couldn't share the same 'channels' 
dimension that they do now.

Update:
Trying to expand the meaning of this now to include the label autoencoding for
anatomically constrained cnn. image_field will translate to inputs, label to outputs,
and we need the aliasing and order of each to be parameters that affect the resizing
and astype calls. We need the two separate, both input and output, cause we're going 
noise the input. Damnit.
'''
class ResizeImagesAndLabels(object):
    def __init__(self, size, 
                 image_field, 
                 label_field,
                 image_alias = True,
                 label_alias = False,
                 image_order = 1,
                 label_order = 0):
        self.size = size
        self.image_field = image_field
        self.image_alias = image_alias
        self.image_order = image_order
        self.label_field = label_field
        self.label_alias = label_alias
        self.label_order = label_order
        
    def __call__(self, data):
        image_entries = []
        label_entries = []

        image_field = self.image_field
        label_field = self.label_field
        
        for image_entry, label_entry in zip(data[image_field], data[label_field]):
            # We can assert here that each pairs at least starts as the same size.
            
            assert np.all(np.asarray(image_entry.shape[1:3]) == np.asarray(label_entry.shape[1:3]))
            
            # Resize the image and label.
            # https://scikit-image.org/docs/dev/api/skimage.transform.html#resize
            
#             print(image_entry.dtype) input was float32
            
            # transpose to go from chan x h x w to h x w x chan and back.
            image_entry = resize(image_entry.transpose([1,2,0]), 
                                 self.size, mode='constant', 
                                 anti_aliasing=self.image_alias,
                                 order = self.image_order, # 1 order default spline interpolation,.
                                 preserve_range=True)
            image_entry = image_entry.transpose([2,0,1])
            
            label_entry = resize(label_entry.transpose([1,2,0]).astype(np.float32), 
                                 self.size, mode='constant', 
                                 anti_aliasing=self.label_alias, # anti-alias applies a blur before downsample. don't want for label.
                                 order=self.label_order, # 0 order for nearest neighbor.
                                 preserve_range=True) 
            label_entry = label_entry.transpose([2,0,1])
            # There's only a single channel in label_entry (with all class types in it), 
            # so let's go ahead and sqeeze for convenience. go from 1 x h x w to h x w.
            label_entry = np.squeeze(label_entry)
            
            # at this point they were float64, not good.  
            # I guess skimage used the native float type.
           
        
            # If order 0, then we assume long type outputs.
            if self.image_order:
                image_entries.append(image_entry.astype(np.float32))
            else:
                image_entries.append(image_entry.astype(np.long))
            
            if self.label_order:
                label_entries.append(label_entry.astype(np.float32))
            else:
                label_entries.append(label_entry.astype(np.long))
        
        # Now, after resizing all the image/label pairs, modify the dictionary accordingly.
        data[self.image_field] = image_entries
        data[self.label_field] = label_entries

        return data
    
'''
Simpler than the above, where we are only concerned with one field.  The above that 
does both images and labels is kind of unnecessary, but oh well. You need the class 
together only if it's like the same random cropping or augmentation you're trying 
to accomplish.
'''
class ResizeTransform(object):
    def __init__(self, size, 
                 field, 
                 alias = True, # alias on the resize, should be False for labels.
                 order = 1 # interpolation order on resize, should be 0 (nearest) for labels.
                ):
        self.size = size
        self.field = field
        self.alias = alias
        self.order = order

    def __call__(self, data):
        entries = []
        field = self.field
        
        for entry in data[field]:

            # transpose to go from chan x h x w to h x w x chan and back.
            entry = resize(entry.transpose([1,2,0]), 
                           self.size, mode='constant', 
                           anti_aliasing=self.alias,
                           order = self.order, # 1 order default spline interpolation,.
                           preserve_range=True)
            entry = entry.transpose([2,0,1])
            
            if self.order:
                entries.append(entry.astype(np.float32))
            else:
                entries.append(entry.astype(np.long))
            
           
        # Now, after resizing, modify the dictionary accordingly.
        data[self.field] = entries

        return data
    
class random_Windower(object):
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
    
    def __call__(self, img):
        im_min, im_max = img.min(), img.max()
        
        # Now scale represents a list of high and low random scale to use.
        # Get the random scale to use in this case.
        sc_ = self.scale[0] + (self.scale[1]-self.scale[0])*np.random.rand()
        # Now basically the same as above.
        im_range = im_max-im_min
        locut = im_min + im_range*((1.0-sc_)*np.random.rand()) # where to start the window.
        hicut = locut+(sc_*im_range)
            
        # We're mapping [locut,hicut] to [im_min,im_max] and clipping outside that range.
        alpha = (img-locut)/(hicut-locut)
        img = (1.0-alpha)*im_min + alpha*im_max
        img = np.clip(img, im_min, im_max)
        
        return img
    
class WindowImagesAndLabels(object):
    '''
    WindowImagesAndLabels: Random intensity windowing augmentation on 
    the input data. Obviously with this augmentation, the labels should
    pass through unmolested. Not sure why I bother to deal with the labels
    then...
    The scales of the windowing represents the
    min and max of how much of the input range is accounted for:
    - default 1.0, 1.0 means no windowing,  
    - e.g., [0.5, 1.0] means pick a random subrange in [.5, 1] of the input 
        range and scale it to the entire input range, then clip/saturate 
        the remaining complement of that subrange.
    '''
    
    def __init__(self, scale=[1.0, 1.0], image_field='images', label_field='labels'):
        # only if scale is scalar.
#         assert scale > 0.0 and scale <= 1.0,\
#             'WindowImagesAndLabels: scale {} must be in (0,1].'.format(scale)

#         scale = sorted(scale)
#         assert min(scale) > 0.0 and max(scale) <= 1.0 and min(scale) <= max(scale),\
#             'random_Windower: scale range {} must be in (0,1].'.format(scale)
        
#         self.scale = scale
        
        self.image_field = image_field
        self.label_field = label_field
        self.windower = random_Windower(scale)
        
    def __call__(self, data):
        image_entries = []
        label_entries = []

        image_field = self.image_field
        label_field = self.label_field
        
        for image_entry, label_entry in zip(data[image_field], data[label_field]):
            # We can assert here that each pairs at least starts as the same size.
#             assert np.all(np.asarray(image_entry.shape[1:3]) == np.asarray(label_entry.shape[1:3]))
            # np.all(np.asarray((256, 256))==np.asarray((256,))) evaluates to true?
#             print('{} and {}'.format(image_entry.shape[1:3], label_entry.shape[1:3]))
            
            # Easy transform worked, let's try the one I wanted.
            # image_entry = np.clip(2.0*image_entry, im_min, im_max)
            
            # This also worked, where it's constant: that is, the [locut,hicut] range always
            # represents the same scale of the input range. scale is just a single number here.
            '''
            im_range = im_max-im_min
            locut = im_min + im_range*((1.0-self.scale)*np.random.rand()) # where to start the window.
            hicut = locut+(self.scale*im_range)
            
            alpha = (image_entry-locut)/(hicut-locut)
            image_entry = (1.0-alpha)*im_min + alpha*im_max
            image_entry = np.clip(image_entry, im_min, im_max)
            '''
            
            
            # collecting this code to the random_Windower class.

#             im_min, im_max = image_entry.min(), image_entry.max()
#             # Now scale represents a list of high and low random scale to use.
#             # Get the random scale to use in this case.
#             sc_ = self.scale[0] + (self.scale[1]-self.scale[0])*np.random.rand()
#             # Now basically the same as above.
#             im_range = im_max-im_min
#             locut = im_min + im_range*((1.0-sc_)*np.random.rand()) # where to start the window.
#             hicut = locut+(sc_*im_range)
            
#             # We're mapping [locut,hicut] to [im_min,im_max] and clipping outside that range.
#             alpha = (image_entry-locut)/(hicut-locut)
#             image_entry = (1.0-alpha)*im_min + alpha*im_max
#             image_entry = np.clip(image_entry, im_min, im_max)
            
            image_entry = self.windower(image_entry)
            
           
            
            image_entries.append(image_entry.astype(np.float32))
            label_entries.append(label_entry.astype(np.long))
        
        # Now, after resizing all the image/label pairs, modify the dictionary accordingly.
        data[self.image_field] = image_entries
        data[self.label_field] = label_entries

        return data
    
class random_GaussNoiser(object):
    def __init__(self, sig_range):
        sig_range = sorted(sig_range)
        assert min(sig_range)>=0.0 and max(sig_range)<=1.0,\
            'random_GaussNoiser: sig_range {} must be in [0.0, 1.0].'.format(sig_range)
        
        self.sig_range = sig_range
    
    def __call__(self, img, sig=None):
        # Keep the noise to the echo cone.
        mask = (img >= np.finfo(np.float32).eps).astype(np.uint)
        
        if not sig:
            # sig_range represents a list of high and low random sigma to use.
            # Get the random scale to use in this case.
            sig = self.sig_range[0] + (self.sig_range[1]-self.sig_range[0])*np.random.rand()
            
        # Make a noise image with sigma, mean 0
        N = sig*np.random.standard_normal(size=img.shape)
        # print('shape of N {}'.format(N.shape))
            
        img = img+N
        img = np.clip(img, 0.0, 1.0)   
        img = (img*mask).astype(np.float32)
        
        return img
        
class random_SpeckleNoiser(object):
    def __init__(self, sig_range):
        sig_range = sorted(sig_range)
        assert min(sig_range)>=0.0 and max(sig_range)<=1.0,\
            'random_GaussNoiser: sig_range {} must be in [0.0, 1.0].'.format(sig_range)
        
        self.sig_range = sig_range
    
    def __call__(self, img, sig=None):
        # Keep the noise to the echo cone.
        mask = (img >= np.finfo(np.float32).eps).astype(np.uint)
        
        if not sig:
            # sig_range represents a list of high and low random sigma to use.
            # Get the random scale to use in this case.
            sig = self.sig_range[0] + (self.sig_range[1]-self.sig_range[0])*np.random.rand()
            
        # Make a noise image with sigma, mean 0
        N = sig*np.random.standard_normal(size=img.shape)
        # print('shape of N {}'.format(N.shape))
            
        img = img+N
        img = np.clip(img, 0.0, 1.0)   
        img = (img*mask).astype(np.float32)
        
        return img
        

class GaussianNoiseEcho(object):
    '''
    GaussianNoise: Add Gaussian noise to the input data. As in the other
    cases, we assume the data given is a dictionary, including a key we're
    provided that leads to a list. We modify each image in that list and 
    return the modified dictionary.
    sig_range is expected to be at most than [0,1]. We generate a sig in the 
    provided range, add said noise to image and clip.
    
    This is echo specific. I don't want to add noise outside of the cone. 
    So I also avoid adding noise to the zero part of the image.
    '''
    def __init__(self, 
                 sig_range=[0.0, 0.0],
                 field='images'):
        
        self.field = field
        self.noiser = random_GaussNoiser(sig_range)
        
    def __call__(self, data):
        entries = []
        
        for entry in data[self.field]:   
            '''
            # Keep the noise to the echo cone.
            mask = (entry >= np.finfo(np.float32).eps).astype(np.uint)
            
            # sig_range represents a list of high and low random sigma to use.
            # Get the random scale to use in this case.
            sig = self.sig_range[0] + (self.sig_range[1]-self.sig_range[0])*np.random.rand()
            
            # Make a noise image with sigma, mean 0
            N = sig*np.random.standard_normal(size=entry.shape)
            # print('shape of N {}'.format(N.shape))
            
            entry = entry+N
            entry = np.clip(entry, 0.0, 1.0)
            
            entry = (entry*mask).astype(np.float32)
            '''
            
            entries.append(self.noiser(entry))
        
        data[self.field] = entries
        
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
    
class RotateImagesAndLabels(object):
    '''
    RotateImagesAndLabels: Random rotation about the ultrasound
    source, assumed located at (0, w/2) in matrix/ij form 
    ((w/2, h) in xy, which is how skimage.transform.rotate
    thinks of it). 
    scale 0 means no rotation.
    scale 5 means a normally distributed random +-15 degrees 
        rotation about the source, maintaining image size and 
        filling with zeros.
    if the random rotation selected is more than 3 std dev out
    it's clipped.
    ref: https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.rotate
    
    Update:
    I'm extending this to be useful in the shape autoencode mode, when images is inputs,
    label is outputs. 
    We want both to undergo the same rotation, but they may not be the same, since the 
    inputs could be noised.
    
    Update:
    Reworked to handle labels/outputs are 2d (N x h x w) or   2d (N x c x h x w)
    '''
    
    def __init__(self, 
                 scalestd=0.0, 
                 rtype='normal', 
                 image_field='images', 
                 label_field='labels',
                 image_order=1,
                 label_order=0):
        # only if scale is scalar.
#         assert scale > 0.0 and scale <= 1.0,\
#             'WindowImagesAndLabels: scale {} must be in (0,1].'.format(scale)
#         assert scalestd >= 0.0 and scalestd <= 60.0,\
#             'RotateImagesAndLabels: scale {} must be in [0,60].'.format(scale)
        
#         assert rtype in ['normal', 'uniform'],\
#             'RotateImagesAndLabels: rtype must be of ["normal", "uniform"]'
        
        self.scale = scalestd
        self.rtype = rtype
        self.image_field = image_field
        self.image_order = image_order
        self.label_field = label_field
        self.label_order = label_order
        self.image_rotater = random_Rotater(scalestd, rtype, image_order)
        self.label_rotater = random_Rotater(scalestd, rtype, label_order)
        
    def __call__(self, data):
        image_entries = []
        label_entries = []

        
        for image_entry, label_entry in zip(data[self.image_field], 
                                            data[self.label_field]):
            
            # print('data size is {}'.format(len(data[self.image_field])))
            
            # Random angle to rotate by.
            # Either +- scale if uniform or +- 3scale if normal.
            if self.rtype == 'normal':
                rot_deg = self.scale*np.random.randn()
                rot_deg = np.clip(rot_deg, -3*self.scale, 3*self.scale)
            else: 
                rot_deg = 2.0*self.scale*np.random.rand() - self.scale

            
            # This debug statment made me find the mistake in the center calculation...
            # print('image_entry shape {}, label_entry shape {}'.format(image_entry.shape, label_entry.shape))
            
            # more debug, found that numpy random number generation is not safe. 
            # Fixed through reseeding workers in the DataLoader 
            # print('rotation: image_entry shape {} by rot {}'.format(image_entry.shape, rot_deg))
            
            # 3 sigma should mean 3 out of 1000 end up being clipped, 
            # hopefully that's not too bad.
            # print('rotating by {} '.format(rot_deg))
            
#             # the rotational center, in col,row format.
#             center = (image_entry.shape[1]/2.0 - 0.5, 
#                       image_entry.shape[0] - 0.5)
            # The above row calculation is a bug. Because the image_entry is usually 1 x h x w, it's 0.5. but it should be 0.5 in any event. (top middle of image)
            
#             # skimage expects channels last
#             # transpose to go from chan x h x w to h x w x chan and back. labels is 2d.
            
#             image_entry = rotate(image_entry.transpose([1,2,0]),
#                                  angle=rot_deg,
#                                  center=center,
#                                  order = self.image_order, # default
#                                  preserve_range=True
#                                 )
#             image_entry = image_entry.transpose([2,0,1])
            
#             # If label is 2d great, but if it's been one-hotted (c x h x w), should still work.
#             if len(label_entry.shape) == 3:
#                 label_entry = rotate(label_entry.transpose([1,2,0]), 
#                                      angle=rot_deg,
#                                      center=center,
#                                      order=self.label_order, # order 0 for nearest neighbor,
#                                      preserve_range=True)
#                 label_entry = label_entry.transpose([2,0,1])
#             else:
#                 label_entry = rotate(label_entry, # already 2d.
#                                      angle=rot_deg,
#                                      center=center,
#                                      order=self.label_order, # order 0 for nearest neighbor,
#                                      preserve_range=True)
            
#             # If order 0, then we assume long type outputs.
#             if self.image_order:
#                 image_entries.append(image_entry.astype(np.float32))
#             else:
#                 image_entries.append(image_entry.astype(np.long))
            
#             if self.label_order:
#                 label_entries.append(label_entry.astype(np.float32))
#             else:
#                 label_entries.append(label_entry.astype(np.long))
            
            image_entries.append(self.image_rotater(image_entry, rot_deg))
            label_entries.append(self.label_rotater(label_entry, rot_deg))
        
        # Now, after resizing all the image/label pairs, modify the dictionary accordingly.
        data[self.image_field] = image_entries
        data[self.label_field] = label_entries

        return data
    
    
class RotateImages(object):
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
                 field='images',
                 order=1):
        # only if scale is scalar.
#         assert scale > 0.0 and scale <= 1.0,\
#             'WindowImagesAndLabels: scale {} must be in (0,1].'.format(scale)
        assert scalestd >= 0.0 and scalestd <= 60.0,\
            'RotateImagesAndLabels: scale {} must be in [0,60].'.format(scale)
        
        assert rtype in ['normal', 'uniform'],\
            'RotateImagesAndLabels: rtype must be of ["normal", "uniform"]'
        
        self.scale = scalestd
        self.rtype = rtype
        self.field = field
        self.order = order
        
    def __call__(self, data):
        entries = []

        field = self.field
        order = self.order
        
        for entry in data[field]:
            
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
            
            # the rotational center, in col,row format.
            center = (entry.shape[1]/2.0 - 0.5, 
                      entry.shape[0] - 0.5)
            
            # skimage expects channels last
            # transpose to go from chan x h x w to h x w x chan and back. labels is 2d.
            
            entry = rotate(entry.transpose([1,2,0]),
                           angle = rot_deg,
                           center= center,
                           order = order, # default 1 order spline
                           preserve_range=True
                          )
            entry = entry.transpose([2,0,1])
            
            
            if order: # order not 0, we want float.
                entries.append(entry.astype(np.float32))
            else:
                entries.append(entry.astype(np.long))
                
        
        # Now, after resizing all the image/label pairs, modify the dictionary accordingly.
        data[self.field] = entries

        return data 
    

class AddSaltPepper(object):
    '''
    Class to augment the inputs with salt and pepper noise with some frequency.
    Assume already normalized image.
    '''
    def __init__(self, 
                 freq = 0.0, 
                 field='inputs',
                 labelCount=2):
        assert freq >= 0.0 and freq <= 1.0,\
            'AddSaltPepper: freq must be in [0,1] ({})'.format(freq)
        
        self.freq = freq
        self.field = field
        self.labelCount = labelCount
        
    def __call__(self, data):
        entries = []
        
        for entry in data[self.field]:
            noise = np.random.randint(0, self.labelCount, 
                                      size=entry.shape).astype(entry.dtype)
            entry = np.where(np.random.random_sample(entry.shape)<=self.freq, 
                             noise, entry)
            
            entries.append(entry)
        
        data[self.field] = entries
        return data
    
class OneHot(object):
    '''
    Class to augment the inputs by converting the channels dimension to onehot.
    Expects 1 x h x w, returns labelCount x h x w
    
    This has to happen after the AddSaltPepper in augmentations I think. The way these
    all behave in composition is confusing. After this there are only two labels, 0 and 1,
    but a non-single channel whose dim reflects the original labelCount...
    
    See for example (and cribbing):
    https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
    '''
    def __init__(self,
                 field='inputs',
                 labelCount = 2,
                 outtype = np.float32):
        self.field = field
        self.labelCount = labelCount
        self.outtype = outtype
    
    # Modified for channels-first from the above link.
    def onehot_initialization_v2(self, a):
        ncols = a.max()+1
        out = np.zeros( (ncols, a.size), dtype=self.outtype)
        out[a.ravel(), np.arange(a.size)] = 1
        out.shape = (ncols,) + a.shape
        return out
        
    def __call__(self, data):
        entries = []
        
        for entry in data[self.field]:
            entry = self.onehot_initialization_v2(entry.squeeze())
            entries.append(entry)
        
        data[self.field] = entries
        return data
    
class identity_Transform(object):
    def __init__(self):
        pass
    
    def __call__(self, data):
        return data
    