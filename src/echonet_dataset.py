import echonet
from echonet.datasets import Echo
import torch

import numpy as np

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset


def random_start_and_end(ed_index, es_index, video_length, length=32):
    """
    Give the ED and ES indexes, length of original video, and length of video clip
    Return a random start and an end index of video clip that covers the ED->ES
    """
    assert es_index - ed_index > 0, "INVALID ED & ES INDICES:\nNo systolic clip found. ES frame comes before ED frame"
    possible_shift = length - (es_index - ed_index + 1)
    if length > video_length:
        # If the length of clip is larger than the length of video
        #  Return the systolic clip instead
        return ed_index, es_index + 1
    elif possible_shift <= 0:
        # Else if possible shift is zero, then the clip starts at ED frame
        start = ed_index
    else:
        # Otherwise, the possible start points are just (max(ED_index - possible shift, 0), min(ED_index, video length - clip length)]
        # Since np.random.randint's range is [low, high) and we want (low, high], add 1 to each side
        start = np.random.randint(max(ed_index - possible_shift + 1, 0), min(video_length - length + 1, ed_index + 1))
    end = start + length
    return start, end


# These echo videos are not properly normalized across RGB channels, 
# and they may have a blue-looking (instead of gray-looking of most other EchoNet Echo Videos)
blue_videos = [89, 105, 325, 653, 721, 739]


def zeroone_normalizer(image_data):
    """
    Zero one normalize the input image data
    Assume the image has the shape of (..., 3)
    """
    norm_data = image_data
    data_shape = norm_data.shape
    norm_data = norm_data.reshape(3, -1)
    norm_data -= np.min(norm_data, axis=1).reshape(3, 1)
    norm_data /= np.max(norm_data, axis=1).reshape(3, 1)
    norm_data = norm_data.reshape(data_shape)
    
    return norm_data


class EchoNetDynamicDataset(Dataset):
    """
    EchoNet Dynamic Dataset wrapper
    image_size: tuple of image size
    norm: name of normalization function
    split: which split of EchoNet-Dynamic dataset will be used
    subset_indices: indices of a subset of dataset
    period: sampling period of the output EchoNet-Dynamic video
    raise_for_es_ed: raise Exception if a clinically denoted diastolic function is found
    """
    def __init__(self,
                 image_size=(112, 112), 
                 clip_length=32, 
                 norm=zeroone_normalizer,
                 split='train',
                 subset_indices=None,
                 period=1,
                 raise_for_es_ed=True,
                 **kwargs
                 ):
        
        mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split=split))
        self.split = split
        self.image_size = image_size
        self.clip_length = clip_length
        self.norm = norm
        self.period = period
        self.raise_for_es_ed = raise_for_es_ed
        # Need filename for saving, and human-selected frames to annotate
        self.echonet_dynamic = echonet.datasets.Echo(split=split,
                                                     target_type=["Filename", "EF", \
                                                                  "LargeIndex", "SmallIndex", \
                                                                  "LargeFrame", "SmallFrame", \
                                                                  "LargeTrace", "SmallTrace"],  
                                                     mean=mean, std=std,  # Normalization
                                                     length=None, max_length=None, period=self.period  # Take all frames
                                                    )
        if subset_indices:
            self.echonet_dynamic = Subset(self.echonet_dynamic, subset_indices)
            
    def __len__(self):
        return len(self.echonet_dynamic)
        
    def __getitem__(self, idx):
        # Get the appropriate info from the Stanford dataset
        video, (filename, EF, es_index, ed_index, es_frame, ed_frame, es_label, ed_label) = self.echonet_dynamic[idx]
        
        es_index //= self.period
        ed_index //= self.period
        
        # This test patient's video has a strange frame rate (too fast)
        if idx == 919 and self.split == 'test':
            factor = 3
            video = torch.Tensor(video).unsqueeze(0)
            video = F.interpolate(video, size=(video.shape[2] * factor, 
                                               video.shape[3], video.shape[4]), 
                                  mode="trilinear", align_corners=True)
            video = video.squeeze(0).numpy()
            ed_index *= factor
            es_index *= factor
        
        if ed_index > es_index and self.raise_for_es_ed:
            print(ed_index, es_index)
            print("ED and ES comes from different heartbeat")
            raise Exception
        
        if self.clip_length == "full":
            video = video
            ed_clip_index = ed_index
            es_clip_index = es_index
        else:
            try:
                start, end = random_start_and_end(ed_index, es_index, video.shape[1], length=self.clip_length)
            except:
                # Rarely the lower bound == higher bound and an Exception will be raised by the NumPy
                # In those cases, use the ed index as the start point of the clip and the es index as the end point
                start = ed_index
                end = es_index + 1
            if end - start < self.clip_length:
                # Interpolate (shrink) the ED to ES video
                video = torch.Tensor(video[:, start:end]).unsqueeze(0)
                video = F.interpolate(video, size=(self.clip_length, self.image_size[0], self.image_size[1]), 
                                      mode="trilinear", 
                                      align_corners=True)
                video = video.squeeze().numpy()
                ed_clip_index = 0
                es_clip_index = self.clip_length - 1
            else:
                video = video[:, start:end]
                ed_clip_index = ed_index - start
                es_clip_index = ed_clip_index + es_index - ed_index
        
        if self.norm:
            video = self.norm(video)
            es_frame = np.squeeze(self.norm(np.expand_dims(es_frame, 0)))
            ed_frame = np.squeeze(self.norm(np.expand_dims(ed_frame, 0)))
            
        if idx in blue_videos and self.split == 'test':
            gray_video = np.dot(video.transpose([1, 2, 3, 0]), np.array([.2989, .5870, .1140])).copy()
            video = np.concatenate([np.expand_dims(gray_video, 0).copy(), 
                                    np.expand_dims(gray_video, 0).copy(), 
                                    np.expand_dims(gray_video, 0).copy()])
        
        return video, (filename, EF, es_clip_index, ed_clip_index, es_index, ed_index, es_frame, ed_frame, es_label, ed_label)


def EDESpairs(diastole, systole):
    diastole = np.sort(np.array(diastole))
    systole = np.sort(np.array(systole))
    clips = []
    
    inds = np.searchsorted(diastole, systole, side='left')
    for i, sf in enumerate(systole):
        if inds[i] == 0: # no prior diastolic frames for this sf
            continue
        best_df = diastole[inds[i]-1] # diastole frame nearest this sf.
        if len(clips) == 0 or best_df != clips[-1][0]:
            clips.append((best_df, sf))
            
    return clips
    
    
def old_random_start_and_end(ED_index, ES_index, video_length, length=32):
    # Deprecated (Not used in the implementation)
    """
         Give the ED and ES indexes, length of original video, and length of video clip
         Return a random start and an end index of video clip that covers the ED->ES
     """
    ED_ES_length = ES_index - ED_index + 1
    if ED_ES_length < 0:
        # If ES frame comes before ED frame
        print("INVALID ED & ES INDICES:\nNo systolic clip found. ES frame comes before ED frame")
        raise Exception
    elif length > video_length:
        # If the length of clip is larger than the length of video
        #  Return the systolic clip instead
        return ED_index, ES_index + 1
    if ED_ES_length > length:
        # If the ED to ES systolic duration is larger than the length of video clip
        # Return the systolic clip instead
        return ED_index, ES_index + 1
    # Otherwise find the largest possible start point
    possible_start = ED_index
    # And maximum possible window shift
    possible_shift = length - ED_ES_length
    
    if possible_shift > possible_start:
        # If possible shift is larger than the number of frames before ED frame
        # Then, the possible start points are in [0, ED_index]
        start = np.random.randint(0, possible_start + 1)
    elif possible_shift <= 0:
        # Else if possible shift is zero, then the clip starts at ED frame
        start = ED_index
    elif possible_shift > (video_length - ES_index - 2):
        # Else if possible shift larger than the number of frames left after the ES frame
        # then reduce the possible shift range so the clip does not exceed the video length
        start = np.random.randint(possible_start + 1 - possible_shift, 
                                  possible_start + 1 - (possible_shift - (video_length - ES_index)) - 3)
    else:
        # Otherwise, the possible start points are just (ED_index - possible shift, ED_index]
        # Since np.random.randint's range is [low, high), add 1 to each side
        start = np.random.randint(possible_start + 1 - possible_shift, possible_start + 1)
        
    num_rest_frames = length - (ES_index - start)
    end = ES_index + num_rest_frames
    
    if end > video_length:
        end = video_length
    
    return start, end