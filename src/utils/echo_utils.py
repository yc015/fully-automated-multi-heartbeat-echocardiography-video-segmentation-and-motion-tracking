'''
Joshua Stough
2/19

Pretty general video and image utilities, used at the moment
for echo loading and visualization.
'''

import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML

import numpy as np
import SimpleITK as itk 
import os

import h5py

from skimage.transform import (resize, 
                               rescale)
from skimage.segmentation import find_boundaries
from numpy.linalg import eig


def transformResizeImage(echo, imsize, outtype=np.float32):
    '''
    Helper function to replace the transform/resize business from 
    normal model training. Given a frames-or-channels x h x w numpy array. 

    Output is channels-first:
    videos come out frames x h x w
    label images come out 1 x h x w 
    particular frames come out 1 x h x w.
    outtype is so that label images can stay integer/long, if called correctly.
    '''
    # Checking dimensions: I'm hoping for frames-or-channels x h x w
    if len(echo.shape) == 2:
        echo = np.expand_dims(echo, axis=0)
    elif len(echo.shape) != 3:
        raise ValueError('readTranformResizeImage: Expected [2,3]-dim, cannot handle'\
                         '{}-dim file {}'.format(len(echo.shape), fname))
    
   
    if outtype == np.float32:
        echo = echo.astype(np.float32)
        # Norm to [0,1], like src/utils/camus_transforms.py
        for frame in range(echo.shape[0]):
            echo[frame] = (echo[frame] - np.min(echo[frame])) / \
                (np.max(echo[frame]) - np.min(echo[frame]))
   
    # Resize, as like code in src/utils/camus_transforms.py
    # resize expects the channels dimension last.
    # This will screw up if echo didn't start channels_first.
    echo = resize(echo.transpose([1,2,0]), 
                  imsize, mode='constant', 
                  anti_aliasing=[False, True][outtype==np.float32], # no anti-alias if label image.
                  order=[0,1][outtype==np.float32]) # use nearest if this is a label image.
    echo = echo.transpose([2,0,1])
    echo = echo.astype(outtype)
    
    return echo

def transformResizeAndFlipImage(echo, imsize, outtype=np.float32):
    '''
    Helper function to replace the transform/resize business from 
    normal model training. Given a frames-or-channels x h x w numpy array.

    Output is channels-first:
    videos come out frames x h x w
    label images come out 1 x h x w 
    particular frames come out 1 x h x w.
    outtype is so that label images can stay integer/long, if called correctly.
    output also flips the w dimension (as all our echos need it).
    '''
    # Checking dimensions: I'm hoping for frames-or-channels x h x w
    if len(echo.shape) == 2:
        echo = np.expand_dims(echo, axis=0)
    elif len(echo.shape) != 3:
        raise ValueError('readTranformResizeImage: Expected [2,3]-dim, cannot handle'\
                         '{}-dim file {}'.format(len(echo.shape), fname))
    
   
    if outtype == np.float32:
        echo = echo.astype(np.float32)
        # Norm to [0,1], like src/utils/camus_transforms.py
        for frame in range(echo.shape[0]):
            echo[frame] = (echo[frame] - np.min(echo[frame])) / \
                (np.max(echo[frame]) - np.min(echo[frame]))
   
    # Resize, as like code in src/utils/camus_transforms.py
    # resize expects the channels dimension last.
    # This will screw up if echo didn't start channels_first.
    echo = resize(echo.transpose([1,2,0]), 
                  imsize, mode='constant', 
                  anti_aliasing=[False, True][outtype==np.float32], # no anti-alias if label image.
                  order=[0,1][outtype==np.float32]) # use nearest if this is a label image.
    echo = echo.transpose([2,0,1])
    echo = echo.astype(outtype)
    
    # And flip
    echo = np.flip(echo, axis=-1).copy()
    
    return echo

def readTransformResizeImage(fname, imsize, outtype=np.float32):
    '''
    Loads an ITK image or video, or numpy image or video file. 

    Output is channels-first:
    videos come out frames x h x w
    label images come out 1 x h x w 
    particular frames come out 1 x h x w.
    outtype is so that label images can stay integer/long, if called correctly.
    
    echo = readTranformResizeImage(echoFilename, image_size)
    '''
    _, file_extension = os.path.splitext(fname)
    
    if file_extension == '.mhd': # An ITK file.
        echoITK = itk.ReadImage(fname)
        # The numpy array returned is in reverse order.
        # That is, from xyz to zyx (frames x height x width)
        echo = itk.GetArrayFromImage(echoITK).astype(np.float32)
    elif file_extension == '.npy':
        echo = np.load(fname)
    else:
        raise Exception('readTranformResizeImage: cannot read '\
                        'file extension: {} of {}'.format(file_extension, fname))
    
    # functional abstraction yay: don't repeat the code.
    return transformResizeImage(echo, imsize, outtype)

def readH5TransformResizeImage(efilename, 
                               dataset_key, 
                               imsize, 
                               outtype=np.float32):
    '''
    readH5TransformResizeImage: what it says.
    '''
    
    with h5py.File(efilename, 'r') as efile:
        echo = np.array(efile[dataset_key])
    
    return transformResizeImage(echo, imsize, outtype)



def makeVideo(arr, cmap=None, saved=False, title=None, interval=2000):
    '''
    makeVideo: given a 3 or 4D array (time x h x w [x 1]), returns an HTML animation 
    of array for viewing in a notebook for example. Cell could say something like:

    %%capture
    # You want to capture the output when the actual call is made. 
    vid = makeVideo(arr, cmap='gray')

    with the following cell just

    vid

    '''
    
    if len(arr.shape) == 4 and arr.shape[-1] == 1: # one channel, otherwise imshow gets confused
        arr = arr.squeeze()
        print('New arr shape {}.'.format(arr.shape))
    
    f, ax = plt.subplots(1,1, figsize=(4,6))
    dispArtist = ax.imshow(arr[0,...], interpolation=None, cmap=cmap)
    f.patch.set_alpha(0.)
    ax.axis("off")
    if title:
        ax.set_title(title)
    
    def updateFig(i):
        # global dispArtist, arr # not sure why I don't need these: 
        # See: https://www.geeksforgeeks.org/global-local-variables-python/
        if i >= arr.shape[0]:
            i = 0

        dispArtist.set_array(arr[i,...])
        return (dispArtist, )
    
    ani = animation.FuncAnimation(f, updateFig, interval=interval/arr.shape[0], 
                                  frames = arr.shape[0], blit = True, repeat = False)
    
    if saved:
        ani = animation.FuncAnimation(f, updateFig, interval=interval/arr.shape[0], 
                                  frames = arr.shape[0], blit = True, repeat = True)
        return ani
    # https://matplotlib.org/api/_as_gen/matplotlib.animation.FuncAnimation.html
    # https://stackoverflow.com/questions/16732379/stop-start-pause-in-python-matplotlib-animation
    # https://stackoverflow.com/questions/43445103/inline-animations-in-jupyter
    # HTML(ani.to_html5_video()) didn't see a big difference in quality
    return HTML(ani.to_jshtml()) # gives a nice button interface for pause and playback.



'''
&nbsp;
## Simpson's Modified Bi-plane Method of Disks
Let's start by writing the top level function, then I can see what the lower level functions actually need to do.

- The a2 and a4 views may have different pixel dimensions. Need to convert to common measure (mm) before combining.
- I'm trying to use some of the code from the [echocv](https://gitlab.com/fornwalt.core/echocv/blob/new_view_segment_pipeline/analyze_segmentations.py) repo Sush put together of the Rahul Deo group's work. That to get the length and axis of the structure we do a linear fit to the coords and project. That idea.

- I'm doing it differently. Right now I get the linear extent as the first principal eigenvector, and the radii as the 
projection onto the 2nd principal component.
'''

'''
Handling nans: this is a great ripped way to interpolate across nans. This could be useful if
the binary volumes below consist of multiple parts. In get2dPucks the radius could be nan if the
center line has no projections within range. I'm not going to use this nan interpolation code yet,
replacing nans with zero instead. But cool code, just in case.
https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
'''
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def computeSimpsonVolume(a2bin, a4bin, a2pix, a4pix):
    '''
    computeSimpsonVolume(a2bin, a4bin, a2pix, a4pix): given binaries of the same structure in 
    A2 and A4 views, and associated pixel dimensions, compute the volume of the structure
    using simpson's biplane method of disks. Basically get a volume by stacking elliptical
    pucks.
    I am assuming someone already took care of the pixel spacing arguments. Like swapped them
    if you went from w x h to h x w in converting the image...
    '''
    # Magic function that gives a length and sequence of radii perpendicular to that length
    L2, R2 = get2dPucks(a2bin, a2pix)
    L4, R4 = get2dPucks(a4bin, a4pix)
    
    # debug:
    # for v, L, R in zip([2,4], [L2, L4], [R2, R4]):
    #     print('computeSimpsonVolume: A{} L(R): {:.2f} ({:.2f}-{:.2f}).'.format(v, L, R.min(), R.max()))
    
    # Now compute volume as the sum of puck volumes. Use the average of L2 and L4 for height.
    # R2 and R4 are numpy arrays. multiply is element-wise.
    # pi * r2 * r4 is pi*A*B area of ellipse, times height. equal shares of height, so
    # (L/10)*pi * ab + (L/10)*pi * cd + ... can factor out the (L/10)*pi
    return ((np.max([L2, L4])*np.pi)/len(R2)) * np.sum(R2*R4)


def get2dPucks(abin, apix, npucks=10, vis=False):
    '''
    get2dPucks(abin, apix): Return the linear extent of the binary structure,
    as well as a sequence of radii about that extent.
    '''
    
    # Empty bin?
    if ~np.any(abin):
        return 1.0, np.zeros((npucks,))
    
    x,y = np.where(abin>0)
    X = np.stack([x,y]) # Coords of all non-zero pixels., 2 x N
    if X.shape[1] < 1: # no pixels, doesn't seem to be a problem usually. 
        return (0.0, np.zeros((npucks,)))
    # Scale dimensions
    X = np.multiply(X, np.array(apix)[:, None]) # Need a broadcastable shape, here 2 x 1
    try:
        val, vec = np.linalg.eig(np.cov(X, rowvar=True))
    except:
        return (0.0, np.zeros((npucks,)))
    
    # Make sure we're in decreasing order of importance.
    eigorder = np.argsort(val)[-1::-1]
    vec = vec[:, eigorder]
    val = val[eigorder]
    
    # Negate the eigenvectors for consistency. Let's say y should be positive eig0,
    # and x should be positive for eig1. I'm not sure if that's what I'm doing here,
    # but just trying to be consistent.
    if vec[0,0] < 0:
        vec[:,0] = -1.0*vec[:,0]
    if vec[1,1] < 0:
        vec[:,1] = -1.0*vec[:,1]
    
    
    mu = np.expand_dims(np.mean(X, axis=1), axis=1)
    
    # Now mu is 2 x 1 mean pixel coord 
    # val is eigenvalues, vec is 2 x 2 right eigvectors (by column), all in matrix ij format
    
    # Use the boundary to get the radii.
    # Project the boundary pixel coords into the eigenspace.
    B = find_boundaries(abin, mode='thick')
    Xb = np.stack(np.where(B))
    Xb = np.multiply(Xb, np.array(apix)[:, None]) # space coords again.
    proj = np.dot((Xb-mu).T,vec) 
    # proj is M x 2, the projections onto 0 and 1 eigs of the M boundary coords.
    
    
    # Now get min max in the first principal direction. That's L! Just L[0] here.
    L_min, L_max = np.min(proj, axis=0), np.max(proj, axis=0)
    L = L_max - L_min
    
    # Partition along the principal axis. The secondary axis represents the radii.
    L_partition = np.linspace(L_min[0], L_max[0], npucks+1)
    
    R = []
    A = np.copy(proj)
    for i in range(len(L_partition)-1):
        # Select those boundary points whose projection on the major axis
        # is within the thresholds. 
        which = np.logical_and(A[:,0] >= L_partition[i],
                               A[:,0] < L_partition[i+1])
        # here which could be empty, if there are multiple components to the binary,
        # which will happen without cleaning for the largest connected component and 
        # such. r will be nan, here I replace with zero.
        # In fact, this math really only works well with nice convex objects.
        if len(which) == 0:
            r = 0
        else:
            r = np.median(np.abs(A[:,1][which]))
        R.append(r)
    
    
    if vis:
        # Some visualization code I didn't know where else to put!
        # B is still in image coords, while mu and the vec and L's are in mm? Use extent.
        # extent = (-0.5, apix[1]*B.shape[1]-0.5, -0.5, apix[0]*B.shape[0]-0.5)# (left, right, bottom, top)
        
        # This got me pretty confused. The issue is that if apix is something other than (1,1), then 
        # B needs to be scaled accordingly. 
        # If apix is significantly less than 1,1, then the 0 order and no anti-aliasing could
        # leave little of the boundary left. Though it would only affect the vis, as the calculation
        # above scaled the boundary points to double, instead of this which returns pixels.
        abin_scaled = rescale(abin, apix, order=0, 
                              preserve_range=True, 
                              anti_aliasing=False, 
                              multichannel=False)
        Bscaled = find_boundaries(abin_scaled, mode='thick')
        
        plt.figure(figsize=(5,5))
        plt.imshow(Bscaled) # , origin='upper', extent=extent)
        
        plt.gca().set_aspect('equal')
        plt.axis('equal')
        

        # Plot the mean and principal projections. But plot needs xy (euclid) instead of ij (matrix)!!!
        # Stupid, keeping the sliced out dimension with None here.
        pca0 = np.array([mu + L_min[0]*vec[:,0, None], mu + L_max[0]*vec[:,0, None]])
        pca1 = np.array([mu + L_min[1]*vec[:,1, None], mu + L_max[1]*vec[:,1, None]])

        # Notice the x and y coord reversed. 
        plt.scatter(x=mu[1], y=mu[0], s=30, marker='*')
        plt.scatter(x=pca0[:,1], y=pca0[:,0], c = [[.2, .4, .2], [.6, .9, .6]]) # Dark green to light green
        plt.scatter(x=pca1[:,1], y=pca1[:,0], c = [[.4, .2, .2], [.9, .6, .6]]) # Dark red to light red

        plt.plot(pca0[:,1], pca0[:,0], 'g--')
        plt.plot(pca1[:,1], pca1[:,0], 'r--')

        for i in range(len(L_partition)-1):
            extent = (L_partition[i]+L_partition[i+1])/2
            points = np.array([mu + extent*vec[:,0, None] - R[i]*vec[:,1, None], # negative radius
                               mu + extent*vec[:,0, None] + R[i]*vec[:,1, None]]) # positive radius
            plt.plot(points[:,1], points[:,0])
            
        
        plt.gca().set_aspect('equal')
        plt.axis('equal')
#         plt.axis('square')

        # title 2d area and approximation.
        plt.suptitle('Actual scaled area {:.2f}, approx {:.2f}'.format(np.prod(apix)*abin.sum(), 
                                                                       (L[0]/npucks)*2*np.sum(R)))
#         plt.tight_layout()
    
    return L[0], np.array(R)