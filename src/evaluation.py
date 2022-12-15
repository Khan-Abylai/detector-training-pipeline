import os

import torch
import torch.nn as nn
import cv2
import numpy as np
import warnings

from models.models import LPDetector
import utils
import tools.bbox_utils as bu
from tools import transforms


def draw_plate_box(img, box, color=(0, 0, 255)):
    cv2.circle(img, (int(box[0]), int(box[1])), 3, color, -1)

    x1 = int((box[0] - box[2] / 2.))
    y1 = int((box[1] - box[3] / 2.))
    x2 = int((box[0] + box[2] / 2.))
    y2 = int((box[1] + box[3] / 2.))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    return img


img_w = 512
img_h = 512
img_size = (img_w, img_h)


model = LPDetector(img_size).cuda()
checkpoint = '../weights/detector_weights_europe.pth'

model = nn.DataParallel(model)
checkpoint = torch.load(checkpoint)['state_dict']
model.load_state_dict(checkpoint)
model.eval()

transform = transforms.DualCompose([
    transforms.ImageOnly(transforms.Transpose()),
    transforms.ImageOnly(transforms.Normalize()),
    transforms.ImageOnly(transforms.ToTensor())
])

total_plate_num = 0
plate_recall = 0

with open('/home/user/mnt/data/EUROPE_ANNOTATION/test.txt') as f:
    ls = f.read().splitlines()
data_dir = '/home/user/mnt/data'
for i, path in enumerate(ls):
    path = os.path.join(data_dir, path)
    img = cv2.imread(path)
    if img is None:
        continue

    h, w, _ = img.shape
    img_for_target = img.copy()
    img_for_pred = img.copy()
    x, _ = transform(cv2.resize(img, img_size))
    x = torch.stack([x]).cuda()
    plate_filename = path.replace('.jpg', '.pb')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plates_target = np.loadtxt(plate_filename).reshape(-1, 12)
    plates_target[..., 0::2] *= w
    plates_target[..., 1::2] *= h

    plate_output = model(x)

    plate_output = plate_output.cpu().detach().numpy()

    rx = float(w) / img_w
    ry = float(h) / img_h

    plates_pred = bu.nms_np(plate_output[0], conf_thres=0.85)
    if len(plates_pred) > 0:
        plates_pred[..., [4, 6, 8, 10]] += plates_pred[..., [0]]
        plates_pred[..., [5, 7, 9, 11]] += plates_pred[..., [1]]
    plates_pred[..., 0::2] *= rx
    plates_pred[..., 1::2] *= ry
    incorrect = False
    for plate_t in plates_target:
        max_iou = 0
        for plate_p in plates_pred:
            max_iou = max(max_iou, bu.bbox_iou_np(np.array([plate_p]), np.array([plate_t]), x1y1x2y2=False))

        total_plate_num += 1
        color = (255, 0, 0)
        if max_iou > 0.7:
            plate_recall += 1
        else:
            color = (0, 255, 255)
            incorrect = True

        img = draw_plate_box(img, plate_t, color=(255, 0, 0))
        img_for_target = draw_plate_box(img_for_target, plate_t, color=color)

    for plate in plates_pred:
        img = draw_plate_box(img, plate, color=(0, 0, 255))
        img_for_pred = draw_plate_box(img_for_pred, plate, color=(0, 0, 255))

    if incorrect:
        cv2.imwrite('/home/user/mnt/debug/exp1/' + str(i) + '_pred.jpg', img_for_pred)
        cv2.imwrite('/home/user/mnt/debug/exp1/' + str(i) + '_target.jpg', img_for_target)


    print(f'Working with {i} th image')

print('plate recall:', plate_recall / total_plate_num)
print('plate num:', total_plate_num)
print('correct plate num:', plate_recall)