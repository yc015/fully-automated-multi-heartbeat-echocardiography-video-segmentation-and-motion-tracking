'''
Joshua Stough, Fornwalt Lab
6/19

Many of the torch examples I'm writing involve identical code 
on batch iteration, training, validation, etc. So I'm putting 
some of that consistent code here.
'''

# Needed modules
import numpy as np
from random import shuffle
import copy
from functools import partial

# Was trying to work with torch multiprocessing, but there is no shared memory thread pool 
# scheme. My BatchIterator and TransformDataset need to have shared memory. 
# https://stackoverflow.com/questions/48822463/how-to-use-pytorch-multiprocessing
# from torch.multiprocessing import Pool # no ThreadPool in torch version
# from torch.multiprocessing import Pool, set_start_method
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass
# from torch.multiprocessing import cpu_count

from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import cpu_count

# torch
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch import nn
from torch.autograd import Variable
from torch.nn import Module
import torch.nn.functional as F

# For torch data ETL.
from torch.utils.data import (DataLoader, Dataset)
from torchvision.transforms import Compose

# for fun
import time
tic, toc = (time.time, time.time)

# for additional segmentation training functionality
from .camus_validate import (dict_extend_values,
                            camus_dice_by_name,
                            cleanupBinary,
                            cleanupSegmentation)




# Class for iterating through data.
class BatchIterator(object):
    '''
    data is key here. data starts as a list, where each element of the list is a dictionary, 
    keyed on the inputs and outputs of the network for that training sample (image/vector/whatever), 
    each with a list value. Inside that list 
    could be for example (in CAMUS) a filename. Then the global_transforms should read the file,
    normalize etc to fit the network input. the augment_transforms get applied at iteration time,
    and can modify the result of the global_transforms to do whatever augmentation needed.
    
    starting with a 'images': [filename(s)] and 'labels': [filename(s)] structure. The transforms 
    take dictionary and update its 'images' or 'labels' or both associated values, such as 
    reading in the filename and replacing each filename with a numpy array, or rotating and scaling
    said numpy array, or adding noise...

    For the simple MNIST example, those global transforms may be empty (the buildMNIST
    already results a finalized form of the list of dictionaries). The augment_transforms show
    the same behavior though in dealing with the data.
    '''
    def __init__(
        self,
        batch_size,
        keys,
        data,
        global_transforms=[],
        augment_transforms=[],
        shuffle=False
    ):
        self.data = copy.deepcopy(data)
        self.keys = keys
        self.length = len(data)
        self.batch_size = batch_size
        self.global_transforms = []
        self.augment_transforms = []
        self.n_batches = int(np.ceil(len(data) / self.batch_size)) # last batch may be undersized.
        self.shuffle = shuffle

        self.global_transforms = global_transforms
        self.augment_transforms = augment_transforms
        
        # We're going to transform the data we've been given, according to the global_transforms)
        transform_helper = partial(self.transform_helper, data=self.data, transforms=self.global_transforms)

        # We're doing it in multi-threaded way so that each thread has the same memory space, 
        # and thus the data dictionary is updated in place. But I'm not sure it's IO bound and so would
        # need multithread (versus process pool, for cpu-bound procs).
        # http://lucasb.eyer.be/snips/python-thread-pool.html
        with Pool(min(6, cpu_count())) as p:
            p.map(transform_helper, range(len(self.data)))
                
    @staticmethod
    def transform_helper(idx, data, transforms):
        for transform in transforms:
            data[idx] = transform(data[idx])
            
    def __len__(self):
        return self.n_batches

    def __iter__(self):
        data = copy.deepcopy(self.data)

        if self.shuffle:
            shuffle(data)
            
        if self.augment_transforms: # empty list here would return False.
            # If there are augment_transforms to be done, then transform the data copy in full.
            # The same transform_helper staticmethod can be used, just with a different set of transforms.
            aug_helper = partial(self.transform_helper, data=data, transforms=self.augment_transforms)
            with Pool(min(6, cpu_count())) as p:
                p.map(aug_helper, range(len(data)))

        for i in range(self.n_batches):
            curr_data = data[i * self.batch_size:np.min([(i+1) * self.batch_size, self.length])]

            # collate batch (from a list of dictionary to a dictionary of lists)
            batch = {}
            for key in self.keys:
                batch[key] = []
                for j in range(len(curr_data)):
                    batch[key].append(curr_data[j][key][0]) # Extra [0] because the single content is in a list...
                batch[key] = np.stack(batch[key]) # stack implicitly creates the batch (0th) dimension to the minibatch 

            yield batch
            
# Better data loading wrappers using pytorch's own Dataset and DataLoader 
# classes
class TransformDataset(Dataset):
    '''
    TransformDataset: Creates a torch dataset out of the provided data, 
    with global_transforms (a Compose object) applied at init, and augment_transforms
    applied when an item is indexed (__getitem__).
    
    If global_transforms loads from disk this can be memory inefficient for large
    datasets. At the end of init all data could be in memory, fyi. Otherwise, you
    could place everything (load, transform, augment) in augment_transforms. 
    
    The input data here could be a list of dictionaries, where each dictionary points
    to input and output for the network, e.g. the path of the image and that of the 
    corresponding segmentation. The global_transforms and augment_transforms can
    then update each dictionary to be the actual numpy input and output or something. 
    See camus_transforms.py for guidance.
    
    See:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    '''
    def __init__(self, 
                 data,
                 global_transforms=None,
                 augment_transforms=None):
        
        super(TransformDataset, self).__init__()
        self.data = copy.deepcopy(data)
        
        if type(global_transforms) == type(list()):
            self.global_transforms = Compose(global_transforms)
        else:
            self.global_transforms = global_transforms
            
        if type(augment_transforms) == type(list()):
            self.augment_transforms = Compose(augment_transforms)
        else:
            self.augment_transforms = augment_transforms
        
        # We're going to transform the data we've been given, according to the global_transforms)
        transform_helper = partial(self.transform_helper, data=self.data, transformComp=self.global_transforms)

        # We're doing it in multi-threaded way so that each thread has the same memory space, 
        # and thus the data dictionary is updated in place. But I'm not sure it's IO bound and so would
        # need multithread (versus process pool, for cpu-bound procs).
        # http://lucasb.eyer.be/snips/python-thread-pool.html
        if self.global_transforms:
            with Pool(processes=min(8, cpu_count())) as p:
                p.map(transform_helper, range(len(self.data)))
    
    @staticmethod
    def transform_helper(idx, data, transformComp):
        data[idx] = transformComp(data[idx])
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the requested sample
        # Remember in this implementation all data could already be in memory. 
        # Otherwise:
        # sample = self.global_transforms(self.data[idx])
        sample = copy.deepcopy(self.data[idx]) # This is a dictionary with two keys
        
        if self.augment_transforms:
            sample = self.augment_transforms(sample)
            
        return sample
    

# Wrap torch's DataLoader for our purposes. First, need a callable collate function 
# for the workers. 
# The input data is a list of dictionaries. need to collate as in BatchIterator
# to a single dictionary with input and output mini-batch for the network.
def torch_collate(data):
    batch = {}
    for key in data[0].keys():
        batch[key] = []
        for j in range(len(data)):
            batch[key].append(data[j][key][0]) # Extra [0] because the single content is in a list...
        batch[key] = np.stack(batch[key]) # stack implicitly creates the batch (0th) dimension to the minibatch 
    return batch
    
    


# Loss as a class, if needed. I thought of all the 
# cooler than entropy losses I wanted. most exist, 
# if the combinations thereof don't.
class BetterLoss(Module):
    def __init__(self, weight=None):
        super(BetterLoss, self).__init__()
        if weight is not None:
            self.weight=torch.from_numpy(weight.astype(np.float32))
            self.stolenLoss = nn.CrossEntropyLoss(weight=self.weight.cuda())
        else:
            self.stolenLoss = nn.CrossEntropyLoss()
    
    def forward(self, input, target):
        # Do really cool stuff.
        # return self.stolenLoss(input, torch.squeeze(target)) 
        return self.stolenLoss(input, target) 
    
    
# Dice loss:
# https://github.com/pytorch/pytorch/issues/1249
# https://github.com/rogertrullo/pytorch/blob/rogertrullo-dice_loss/torch/nn/functional.py#L708
class DiceLoss(Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        self.weight = None
        if weight is not None:
            self.weight=torch.from_numpy(np.array(weight).astype(np.float32))
    
    def forward(self, input, target):
        """
        def dice_loss(input,target):
        input is a torch variable of size Batch x nclasses x H x W representing log probabilities for each class
        target is a 1-hot representation of the groundtruth, shoud have same size as the input
        """
        assert input.size() == target.size(), "Input sizes must be equal."
        assert input.dim() == 4, "Input must be a 4D Tensor."
        # We'll assume I've coded this to where we don't need this check constantly.
#         uniques=np.unique(target.cpu().numpy())
#         assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

        probs=F.softmax(input, dim=1)
        num=probs*target#b,c,h,w--p*g
        num=torch.sum(num,dim=3)#b,c,h
        num=torch.sum(num,dim=2)


        den1=probs*probs#--p^2
        den1=torch.sum(den1,dim=3)#b,c,h
        den1=torch.sum(den1,dim=2)


        den2=target*target#--g^2
        den2=torch.sum(den2,dim=3)#b,c,h
        den2=torch.sum(den2,dim=2)#b,c


        dice=2*(num/(den1+den2))
        dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

        dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

        return dice_total


    '''  
# Training and Validation Loops 


As in the VNET example still, we're going to run training and validation loops. The training loop is interesting as it includes the torch ability to manage effective minibatch sizes in smaller runs. That way, even if you run out of memory with batchsize 32, then we can push it through 16 at a times, run 2 times while the gradients accumulate, then do the backprop.  Cool. 

Again, more information in the [VNET](nvidiaDLI_VNetExample/VNET.ipynb) example.

Bit of confusion on the zero_grad, backward and step. Following the code example on [this discussion](https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20)

- typically do zero, backward, step every iteration.
- for less memory, always backward, occasionally step and zero.  This accumulates the gradients for a larger effective batch size.

But the example from VNET does a weirder process, occasionally zero but always backward and step. That seems to use some gradient info more than once. It doesn't seem to make much difference (still getting .94+/.92ish), but I'm going to rewrite to the way that makes sense to me, the second way above. Also see [torch optim](https://pytorch.org/docs/stable/optim.html) documentation.

Rather than have a run_seg_training and run_ae_training, where most of the code is equivalent, I'm merging the two. That's a little weird, since the autoencoder doesn't do dice and cleaning like the segmentation task may.
'''
            
# Run training for one epoch.
def run_training(network,  
                 data_iterator,
                 effective_batchsize,
                 criterion = BetterLoss(),
                 optimizer = None,
                 cur_learning_rate=1e-3,
                 cur_weight_decay=1e-5,
                 keys = ['inputs', 'outputs'],
                 do_dice = False,
                 do_cleaning = False):
    '''
    Loop over the data_iterator once and optimize the weights of the network.
    '''
    
    in_key, out_key = keys
    
    # Instantiate an optimizer and criterion
    network.train()
    
    # reinit seed for augmentation
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-404232359
    np.random.seed()
    
    # Borrowing from: https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(net_ae.parameters(), 
    #                             lr=cur_learning_rate,
    #                             weight_decay=cur_weight_decay)
    if not optimizer:
        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=cur_learning_rate,
                                     weight_decay=cur_weight_decay)
        
    running_loss = 0.0
    
    if do_dice:
        dices = {}
    
    
    for i, data in enumerate(data_iterator, 1):
        
        # We could assert on our inputs and outputs here, for sanity
        if len(data[out_key].shape) == 4 and data[out_key].shape[1] != 1: # has been one-hotted, so make sure only [0,1]
            uniques=np.unique(data[out_key])
            assert set(list(uniques))<=set([0,1]), 'run_training: need [0,1] one-hot output only.'
        
        # get the inputs
        inputs, correct_outputs = torch.from_numpy(data[in_key]), torch.from_numpy(data[out_key])
        
        # print('run_training: inputs size {}, correct_outputs size {}'.format(inputs.size(), correct_outputs.size()))

        if torch.cuda.is_available:
            inputs, correct_outputs = inputs.cuda(), correct_outputs.cuda()

        # wrap them in Variable
        inputs, correct_outputs = Variable(inputs), Variable(correct_outputs)
        
        
        # get network output
        net_outputs = network(inputs)
        
        # This sigmoid is kind of advised and kind of not, it somewhat mutes the
        # really confident (pos or neg) scores coming out of the network.
        # It makes training a little slower, but also a little safer maybe?
        # https://stats.stackexchange.com/questions/163695/non-linearity-before-final-softmax-layer-in-a-convolutional-neural-network
        # I'm moving this whole thing to the network itself.
        # net_outputs = torch.nn.Sigmoid()(net_outputs)

        net_loss = criterion(net_outputs, correct_outputs)

        # https://discuss.pytorch.org/t/freezing-the-updates-without-freezing-the-gradients/7358
        # This discussion on a more complicated topic talks about how it's the loss object that 
        # updates the network gradients when you call backward. But how does the loss (criterion)
        # even know what the network parameters are? It must be hidden in net_outputs...
        net_loss.backward()
        
        # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903
        # The gradients accumulate without the call to zero_grad. So zeroing only occasionally means you're
        # accumulating an effectively larger batch size.
        if (i % effective_batchsize) == 0:
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()
        

        running_loss += net_loss.detach().cpu().item()
        
        
        net_outputs = net_outputs.detach().cpu().numpy()
            
        if do_cleaning:
            net_outputs = cleanupSegmentation(net_outputs)

        if do_dice:
            dices = dict_extend_values(dices, 
                                       camus_dice_by_name(net_outputs,
                                                          data[out_key]))


    # Ready to return info, including loss and example.
    avg_loss = running_loss / len(data_iterator)
    one_output = net_outputs[0]    
    one_input = inputs.detach().cpu().numpy()[0]
    one_correct_output = correct_outputs.detach().cpu().numpy()[0]
    
    if do_dice: # return dices too.
        return avg_loss, one_output, one_input, one_correct_output, dices
    
    return avg_loss, one_output, one_input, one_correct_output


''' Former, silly way of writing the dice and cleaning logic.
        # If we're doing dice
        if do_dice:
            if do_cleaning:
                net_outputs = cleanupSegmentation(net_outputs.detach().cpu().numpy())
            else:
                net_outputs = net_outputs.detach().cpu().numpy()

            dices = dict_extend_values(dices, 
                                       camus_dice_by_name(net_outputs,
                                                          data[out_key]))
        
    # Ready to return info, including loss and example.
    avg_loss = running_loss / len(data_iterator)
    
    if do_dice:
        one_output = net_outputs[0]
    else:
        one_output = net_outputs.detach().cpu().numpy()[0]
        
    one_input = inputs.detach().cpu().numpy()[0]
    one_correct_output = correct_outputs.detach().cpu().numpy()[0]
    
    if do_dice: # return dices too.
        return avg_loss, one_output, one_input, one_correct_output, dices
    
    return avg_loss, one_output, one_input, one_correct_output
'''

# run validation 
def run_validation(network,  
                   data_iterator,
                   criterion = BetterLoss(),
                   keys = ['inputs', 'outputs'],
                   do_dice = False,
                   do_cleaning = False):
    '''
    Loop over the data_iterator once witout optimizing. compute the loss.
    Almost too similar to run_training, but oh well.
    '''
    
    in_key, out_key = keys
    
    # Prevent weight updates.
    network.eval()
    
    # reinit seed for augmentation. if no augment, won't matter.
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-404232359
    np.random.seed()
    
    # criterion = nn.MSELoss()
    
    running_loss = 0.0
    
    if do_dice:
        dices = {}
    
    with torch.no_grad():
        for i, data in enumerate(data_iterator, 1):
            # get the inputs
            inputs, correct_outputs = torch.from_numpy(data[in_key]), torch.from_numpy(data[out_key])

            if torch.cuda.is_available:
                inputs, correct_outputs = inputs.cuda(), correct_outputs.cuda()

            # wrap them in Variable
            inputs, correct_outputs = Variable(inputs), Variable(correct_outputs)


            # get network output
            net_outputs = network(inputs)

            net_loss = criterion(net_outputs, correct_outputs)

            running_loss += net_loss.detach().cpu().item()
            
            
            
            net_outputs = net_outputs.detach().cpu().numpy()
            
            if do_cleaning:
                net_outputs = cleanupSegmentation(net_outputs)

            if do_dice:
                dices = dict_extend_values(dices, 
                                           camus_dice_by_name(net_outputs,
                                                              data[out_key]))


    # Ready to return info, including loss and example.
    avg_loss = running_loss / len(data_iterator)
    one_output = net_outputs[0]    
    one_input = inputs.detach().cpu().numpy()[0]
    one_correct_output = correct_outputs.detach().cpu().numpy()[0]
    
    if do_dice: # return dices too.
        return avg_loss, one_output, one_input, one_correct_output, dices
    
    return avg_loss, one_output, one_input, one_correct_output


# run validation and return all info. 
def run_validation_returnAll(network,  
                             data_iterator,
                             keys = ['inputs', 'outputs'],
                             infokey = 'pat_view_phase',
                             do_dice = True,
                             do_cleaning = False,
                             require_out = False):
    '''
    Loop over the data_iterator once witout optimizing. Record 
    all outputs along with the corresponding infokey information
    from the batches.
    Very similar to other code, but I needed the patient/view/phase 
    info to follow along. 
    '''
    
    in_key, out_key = keys
    
    # Prevent weight updates.
    network.eval()
    
    # reinit seed for augmentation
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-404232359
    np.random.seed()

    
    if do_dice:
        dices = {}
        
    net_outputs_SoFar = np.array([])
    infokey_SoFar = np.array([])
    
    with torch.no_grad():
        for i, data in enumerate(data_iterator, 1):
            # get the inputs
            # We're not actually using correct_outputs, cause we're not computing loss here.
            # inputs, correct_outputs = torch.from_numpy(data[in_key]), torch.from_numpy(data[out_key])
            inputs = torch.from_numpy(data[in_key])

            if torch.cuda.is_available():
                # inputs, correct_outputs = inputs.cuda(), correct_outputs.cuda()
                inputs = inputs.cuda()
            else:
                print('Did not FIND CUDA?!!?\n')
                

            # wrap them in Variable 
            # inputs, correct_outputs = Variable(inputs), Variable(correct_outputs)
            inputs = Variable(inputs)


            # get network output
            net_outputs = network(inputs)

            if require_out:
                net_outputs = net_outputs["out"].detach().cpu().numpy()
            else:
                net_outputs = net_outputs.detach().cpu().numpy()
            
            if do_cleaning:
                net_outputs = cleanupSegmentation(net_outputs)

            if do_dice:
                dices = dict_extend_values(dices, 
                                           camus_dice_by_name(net_outputs,
                                                              data[out_key]))
                
            # Now cat this part with previous parts
            if len(net_outputs_SoFar) == 0:
                net_outputs_SoFar = net_outputs.copy()
                infokey_SoFar = data[infokey].copy()
            else:
                net_outputs_SoFar = np.concatenate([net_outputs_SoFar, net_outputs], axis=0)
                infokey_SoFar = np.concatenate([infokey_SoFar, data[infokey]], axis=0)

    if do_dice:
        return net_outputs_SoFar, infokey_SoFar, dices
    
    return net_outputs_SoFar, infokey_SoFar




'''
Functions for training the anatomically-constrained CNN.
'''

'''
Looks like ripped from OneHot in [camus_tranforms](./src/utils/camus_transforms.py). But I need it here, as it doesn't make a ton of sense in my head to include a separate key in the dataset, that does what happens to the labels but then also one_hots...
'''
class OneHotBatch(object):
    '''
    Class to augment the inputs by converting the channels dimension to onehot.
    Expects N x 1 x h x w , or N x h x w also works. 
    
    https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
    '''
    def __init__(self,
                 labelCount = 4,
                 outtype = np.float32):
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
        # squeeze is not nec for h x w,, but helps if 1 x h x w for each sample in the batch. 
        return np.stack([self.onehot_initialization_v2(entry.squeeze()) for entry in data])
    

# Run training for one epoch.
def run_training_ACNN(network,
                      autoencoder,
                      data_iterator,
                      effective_batchsize,
                      prior_weight = 1e-2,
                      cur_learning_rate=1e-3,
                      cur_weight_decay=1e-5,
                      keys = ['images', 'labels'],
                      do_dice = False,
                      do_cleaning = False):
    '''
    Loop over the data_iterator once and optimize the weights of the network.
    '''
    
    in_key, out_key = keys
    
    one_hotter = OneHotBatch()
    
    # This won't screw up the .requires_grad we've initialized.
    # https://discuss.pytorch.org/t/model-train-and-requires-grad/25845
    network.train()
    
    # reinit seed for augmentation
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-404232359
    np.random.seed()
    
    
    # Instantiate an optimizer and criterion
    
    # Borrowing from: https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(net_ae.parameters(), 
    #                             lr=cur_learning_rate,
    #                             weight_decay=cur_weight_decay)
    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=cur_learning_rate,
                                 weight_decay=cur_weight_decay)
    
    
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    
        
    running_loss = 0.0
    
    if do_dice:
        dices = {}
    
    
    for i, data in enumerate(data_iterator, 1):
        # get the inputs
        inputs, correct_outputs, correct_one_hots = \
            torch.from_numpy(data[in_key]), \
            torch.from_numpy(data[out_key]), \
            torch.from_numpy(one_hotter(data[out_key]))

        if torch.cuda.is_available:
            inputs, correct_outputs, correct_one_hots = \
                inputs.cuda(), correct_outputs.cuda(), correct_one_hots.cuda()

        # wrap them in Variable
        # inputs, correct_outputs, correct_one_hots = \
        #    Variable(inputs), Variable(correct_outputs), Variable(correct_one_hots)
        
        inputs, correct_outputs = \
            Variable(inputs), Variable(correct_outputs)
        
        
        with torch.no_grad():
            correct_encodings = autoencoder(correct_one_hots, -1)
        
        
        # get network output
        net_outputs, net_encodings = network(inputs)
        


        ce_loss = criterion1(net_outputs, correct_outputs)
        ae_loss = criterion2(net_encodings, correct_encodings)
        
        loss = ce_loss + prior_weight*ae_loss
        
        # print('losses ce {:.4f}  and ae {:.4f}.'.format(ce_loss, ae_loss))

        # https://discuss.pytorch.org/t/freezing-the-updates-without-freezing-the-gradients/7358
        # This discussion on a more complicated topic talks about how it's the loss object that 
        # updates the network gradients when you call backward. But how does the loss (criterion)
        # even know what the network parameters are? It must be hidden in net_outputs...
        loss.backward()
        
        # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903
        # The gradients accumulate without the call to zero_grad. So zeroing only occasionally means you're
        # accumulating an effectively larger batch size.
        if (i % effective_batchsize) == 0:
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()
        

        running_loss += loss.detach().cpu().item()
        
        
        net_outputs = net_outputs.detach().cpu().numpy()
            
        if do_cleaning:
            net_outputs = cleanupSegmentation(net_outputs)

        if do_dice:
            dices = dict_extend_values(dices, 
                                       camus_dice_by_name(net_outputs,
                                                          data[out_key]))


    # Ready to return info, including loss and example.
    avg_loss = running_loss / len(data_iterator)
    one_output = net_outputs[0]    
    one_input = inputs.detach().cpu().numpy()[0]
    one_correct_output = correct_outputs.detach().cpu().numpy()[0]
    
    if do_dice: # return dices too.
        return avg_loss, one_output, one_input, one_correct_output, dices
    
    return avg_loss, one_output, one_input, one_correct_output


# run validation 
def run_validation_ACNN(network,  
                       autoencoder,
                       data_iterator,
                       prior_weight = 1e-2,
                       keys = ['images', 'labels'],
                       do_dice = False,
                       do_cleaning = False):
    '''
    Loop over the data_iterator once witout optimizing. compute the loss.
    Almost too similar to run_training, but oh well.
    '''
    
    in_key, out_key = keys
    
    one_hotter = OneHotBatch()
    
    # Prevent weight updates.
    network.eval()
    
    # reinit seed for augmentation, if no augment won't matter.
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-404232359
    np.random.seed()
    
    # criterion = nn.MSELoss()
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    
    running_loss = 0.0
    
    if do_dice:
        dices = {}
    
    with torch.no_grad():
        for i, data in enumerate(data_iterator, 1):
            # get the inputs
            inputs, correct_outputs, correct_one_hots = \
                torch.from_numpy(data[in_key]), \
                torch.from_numpy(data[out_key]), \
                torch.from_numpy(one_hotter(data[out_key]))

            if torch.cuda.is_available:
                inputs, correct_outputs, correct_one_hots = \
                    inputs.cuda(), correct_outputs.cuda(), correct_one_hots.cuda()

            # wrap them in Variable
            inputs, correct_outputs = \
                Variable(inputs), Variable(correct_outputs)


            with torch.no_grad():
                correct_encodings = autoencoder(correct_one_hots, -1)


            # get network output
            net_outputs, net_encodings = network(inputs)



            ce_loss = criterion1(net_outputs, correct_outputs)
            ae_loss = criterion2(net_encodings, correct_encodings)

            loss = ce_loss + prior_weight*ae_loss

            # print('val losses ce {:.4f}  and ae {:.4f}.'.format(ce_loss, ae_loss))

            

            running_loss += loss.detach().cpu().item()
            
            net_outputs = net_outputs.detach().cpu().numpy()

            if do_cleaning:
                net_outputs = cleanupSegmentation(net_outputs)

            if do_dice:
                dices = dict_extend_values(dices, 
                                           camus_dice_by_name(net_outputs,
                                                              data[out_key]))


    # Ready to return info, including loss and example.
    avg_loss = running_loss / len(data_iterator)
    one_output = net_outputs[0]    
    one_input = inputs.detach().cpu().numpy()[0]
    one_correct_output = correct_outputs.detach().cpu().numpy()[0]
    
    if do_dice: # return dices too.
        return avg_loss, one_output, one_input, one_correct_output, dices
    
    return avg_loss, one_output, one_input, one_correct_output
