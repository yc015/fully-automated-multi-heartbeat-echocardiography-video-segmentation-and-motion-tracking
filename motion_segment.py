from src.model.R2plus1D_18_MotionNet import R2plus1D_18_MotionNet
from src.echonet_dataset import zeroone_normalizer
from src.fuse_utils import segment_a_video_with_fusion, compute_ef_using_putative_clips
from src.visualization_utils import make_annotated_gif
import echonet

import torch
import torch.nn as nn

import cv2
import numpy as np

import argparse
import sys
import pickle
import os

ap = argparse.ArgumentParser(description="Segment and motion track heart structure in an Echo Video")

# Video path
ap.add_argument("-p", "--path", required=True, type=str, help="Path to the video")

# model path
ap.add_argument("-m", "--model", required=False, type=str,
                help="Path to the saved model weights",
                default="save_models/R2plus1DMotionSegNet_model.pth")

# Device
ap.add_argument("-d", "--device", required=False, type=str,
                help="Which device to use: CPU or GPU",
                default="cpu")

# Fuse method
ap.add_argument("--fuse_method", required=False, type=str, help="Fuse method",
                default="simple")

# Number of shifts for fusion augmentation
# n == 1 No fusion but use the original video
# n >  1 Shift video by n times and fuse n segmentation results
ap.add_argument("-f", "--fuse", required=False, type=int, help="Number of shifted video clips to fuse",
                default=1)

# fuse step. How far does the video shift each time
ap.add_argument("-s", "--step", required=False, type=int, help="Step of shifting",
                default=1)

# Path to the output files
ap.add_argument("-o", "--output", required=False, type=str,
                help="Path to the output files",
                default=".")

# Verbosity
ap.add_argument("-v", "--verbose", required=False, type=bool, help="Verbosity",
                default=True)

ap.add_argument("-c", "--content", required=False, type=str, help="Content of the output: gif, binary, binary_video, all",
                default="binary")

# Prin

args = ap.parse_args()

model_save_path = args.model

model = torch.nn.DataParallel(R2plus1D_18_MotionNet())
model.to(args.device)
torch.cuda.empty_cache()
model.load_state_dict(torch.load(model_save_path)["model"])
if args.verbose:
    print(f'R2+1D MotionNet has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.')

model.eval();

# See https://github.com/echonet/dynamic/blob/108518f305dbd729c4dca5da636601ebd388f77d/echonet/utils/__init__.py#L33
# and See https://stackoverflow.com/a/42166299
video = echonet.utils.loadvideo(args.path).astype(np.float32)

video = zeroone_normalizer(video)

class_list = [0, 1]

segmentations = segment_a_video_with_fusion(video, model=model, interpolate_last=True, 
                                            step=args.step, num_clips=args.fuse,
                                            fuse_method=args.fuse_method, class_list=class_list)

predicted_efs, edes_pairs = compute_ef_using_putative_clips(segmentations, test_pat_index=args.path,
                                                            return_edes=True)

if args.verbose:
    print("Identified {:d} systoles".format(len(predicted_efs)))
    if len(predicted_efs) > 0:
        print("\nEjection fractions measured at each systole are:")
        for i in range(len(predicted_efs)):
            print("Systole #{:d}: ED {:d} & ES {:d} length={:d}".format(i + 1, edes_pairs[i][0], edes_pairs[i][1],
                                                                        edes_pairs[i][1] - edes_pairs[i][0]))
            print("EF: {:.2f}\n".format(predicted_efs[i]))
        print("The average ejection fraction is {:.2f}".format(np.mean(predicted_efs)))

filename = args.path[args.path.rfind("/") + 1:args.path.rfind(".")]

content = args.content.lower().split(",")

if "gif" in content or "all" in content:
    make_annotated_gif(segmentations, video, filename= os.path.join(args.output, filename + "_annotated.gif"))

if "binary" in content or "all" in content:
    for i in range(len(edes_pairs)):
        ed_index = edes_pairs[i][0]
        es_index = edes_pairs[i][1]
        
        with open(os.path.join(args.output, filename + "_ED_Frame_{:d}_segmentation.pkl".format(ed_index)), "wb") as outfile:
            pickle.dump(segmentations[ed_index], outfile)
        outfile.close()
        
        with open(os.path.join(args.output, filename + "_ES_Frame_{:d}_segmentation.pkl".format(es_index)), "wb") as outfile:
            pickle.dump(segmentations[es_index], outfile)
        outfile.close()

if "binary_video" in content or "all" in content:
    with open(os.path.join(args.output, filename + "_whole_video_segmentation.pkl"), "wb") as outfile:
        pickle.dump(segmentations, outfile)
    outfile.close()
    
    