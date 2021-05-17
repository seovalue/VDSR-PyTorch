# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse

import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
import os
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import pdb
import math
import numpy as np

from vdsr_pytorch import VDSR

parser = argparse.ArgumentParser(description="Accurate Image Super-Resolution Using Very Deep Convolutional Networks")

parser.add_argument("--directory", type=str,
                    help="Directory where features are saved.")
parser.add_argument("--datapath", type=str)                    
parser.add_argument("--feature-type", type=str)
# parser.add_argument("--file", type=str,
#                     help="Test low resolution image name.")
parser.add_argument("--scale-factor", default=4, type=int, choices=[2, 3, 4],
                    help="Super resolution upscale factor. (default:4)")
parser.add_argument("--weights", type=str,
                    help="Generator model name.")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")

args = parser.parse_args()
print(args)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if args.cuda else "cpu")

# Crop features

root_dir = args.directory + args.feature_type +"/"
# file_list = os.listdir(root_dir)

data_path = args.datapath
datatype = args.feature_type
rescale_factor = args.scale_factor
if not os.path.exists(data_path):
    raise Exception(f"[!] {data_path} not existed")

hr_path = os.path.join(data_path, 'LR_2')
hr_path = os.path.join(hr_path, datatype)
print(hr_path)
hr_path = sorted(glob(os.path.join(hr_path, "*.*")))
hr_imgs = []
w, h = Image.open(hr_path[0]).size
width = int(w / 16)
height = int(h / 16)
lwidth = int(width / rescale_factor)
lheight = int(height / rescale_factor)
print("lr: ({} {}), hr: ({} {})".format(lwidth, lheight, width, height))
for hr in hr_path:
    hr_image = Image.open(hr)  # .convert('RGB')\
    for i in range(16):
        for j in range(16):
            (left, upper, right, lower) = (
            i * width, j * height, (i + 1) * width, (j + 1) * height)
            crop = hr_image.crop((left, upper, right, lower))
            hr_imgs.append(crop)

# create model
model = VDSR(args.scale_factor).to(device)

# Load state dicts
model.load_state_dict(torch.load(args.weights, map_location=device))

# Set eval mode
model.eval()

count = 0
for file in hr_imgs:
    # Open feature
    feature = Image.open(file)
    feature_width = int(feature.size[0] * args.scale_factor)
    feature_height = int(feature.size[1] * args.scale_factor)
    feature = feature.resize((feature_width, feature_height), Image.BICUBIC)
    y = feature.split()[0]


    preprocess = transforms.ToTensor()
    inputs = preprocess(y).view(1, -1, y.size[1], y.size[0])

    inputs = inputs.to(device)

    with torch.no_grad():
        out = model(inputs)
    out = out.cpu()
    out_image_y = out[0].detach().numpy()
    out_image_y *= 255.0
    out_image_y = out_image_y.clip(0, 255)
    out_image_y = Image.fromarray(np.uint8(out_image_y[0]), mode="L")

    out_img = Image.merge("Y", [out_image_y])
    out_img.save(f"outputs/outputs_{count}_{args.scale_factor}x.png")
    count += 1
