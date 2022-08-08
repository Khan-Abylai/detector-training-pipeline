import argparse
import torch
import torch.nn as nn
import numpy as np

from models.models import LPDetector

parser = argparse.ArgumentParser()
parser.add_argument('--weights_path', type=str, default='/mnt/workspace/model_dir_lp_detector/23_500_TRAIN_|_Plates_0.000_575632__Loss_0.082_VAL_|_Plates_Recall_0.979_60294_Val_loss_0.085,_lr=3.90625e-05.pth')
parser.add_argument('--out_path', type=str, default='/mnt/workspace/extracted_weights/detector_weights.np')
parser.add_argument('--img_w', type=int, default=512)
parser.add_argument('--img_h', type=int, default=512)
args = parser.parse_args()

model = LPDetector((args.img_w, args.img_h)).cuda()
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
