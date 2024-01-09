import argparse
import torch
import torch.nn as nn
import numpy as np
import config
from models.models import LPDetector

parser = argparse.ArgumentParser()
parser.add_argument('--weights_path', type=str, default='/mnt/data/detector/weights/detector_wnpr.pth')
parser.add_argument('--out_path', type=str, default='../weights/wnpr_detector_weights.np')
parser.add_argument('--img_w', type=int, default=config.IMG_W)
parser.add_argument('--img_h', type=int, default=config.IMG_W)
args = parser.parse_args()

model = LPDetector((args.img_w, args.img_h))
model = nn.DataParallel(model)
checkpoint = torch.load(args.weights_path)['state_dict']
model.load_state_dict(checkpoint)
model.eval()
model.cpu()

s_dict = model.state_dict()
total = 0
t = 'num_batches_tracked'
np_weights = np.array([], dtype=np.float32)
for k, v in s_dict.items():
    if k[-len(t):] == t:
        continue
    total += v.numel()
    v_reshape = v.reshape(-1)
    np_v = v_reshape.data.numpy()
    np_weights = np.concatenate((np_weights, np_v))

print(total)
print(np_weights.shape)
print(np_weights)
print(np_weights.dtype)

np_weights.tofile(args.out_path)
