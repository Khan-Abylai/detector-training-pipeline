import os
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn as nn

import tools.bbox_utils as bu
from models.models import LPDetector
from tools import transforms
import config

img_w = config.IMG_W
img_h = config.IMG_H
img_size = (img_w, img_h)
model = LPDetector(img_size).cuda()
checkpoint_name = '/home/user/src/weights/detector_base.pth'
model = nn.DataParallel(model)
checkpoint = torch.load(checkpoint_name)['state_dict']
model.load_state_dict(checkpoint)
model.eval()

dummy_input = torch.randn(1, 3, img_w, img_h).cuda()
input_names = ["actual_input"]
output_names = ["output"]

if isinstance(model, torch.nn.DataParallel):
    model = model.module

torch.onnx.export(model, dummy_input, checkpoint_name.replace('.pth', '.onnx'), verbose=True, input_names=input_names,
                  output_names=output_names, export_params=True, opset_version=11)
print("Model converted to the onnx")