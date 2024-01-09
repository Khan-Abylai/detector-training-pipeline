import os
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn as nn

import config
import tools.bbox_utils as bu
from models.models import LPDetector
from tools import transforms

img_w = config.IMG_W
img_h = config.IMG_H
img_size = (img_w, img_h)
model = LPDetector(img_size).cuda()

base_folder = '/home/user/detector_pipeline'

checkpoint = os.path.join(base_folder, 'weights/detector_weights_new_uae.pth')
model = nn.DataParallel(model)
checkpoint = torch.load(checkpoint)['state_dict']
model.load_state_dict(checkpoint)
model.eval()

transform = transforms.DualCompose(
    [transforms.ImageOnly(transforms.Transpose()), transforms.Normalize(), transforms.ToTensor()])

ls = glob(os.path.join(base_folder, 'data/uae_data/*'))

for image_path in ls:
    img = cv2.imread(image_path)
    img_orig = img.copy()
    img = cv2.resize(img, img_size)
    fake_bboxes = np.random.rand(1, 12)
    x, _, = transform(img, fake_bboxes)
    x = torch.stack([x]).cuda()

    plate_output = model(x)
    plate_output = plate_output.cpu().detach().numpy()
    rx = float(img_orig.shape[1]) / img_w
    ry = float(img_orig.shape[0]) / img_h
    plates = bu.nms_np(plate_output[0], conf_thres=0.85)
    extension = os.path.basename(image_path).split('.')[-1]
    if len(plates) > 0:
        plates[..., [4, 6, 8, 10]] += plates[..., [0]]
        plates[..., [5, 7, 9, 11]] += plates[..., [1]]

        for plate_idx, plate in enumerate(plates):
            cv2.circle(img_orig, (int(plate[0] * rx), int(plate[1] * ry)), 1, (0, 255, 255), -1)

            cv2.circle(img_orig, (int((plate[4]) * rx), int((plate[5]) * ry)), 1, (0, 255, 0), -1)
            cv2.circle(img_orig, (int((plate[6]) * rx), int((plate[7]) * ry)), 1, (0, 255, 0), -1)
            cv2.circle(img_orig, (int((plate[8]) * rx), int((plate[9]) * ry)), 1, (0, 255, 0), -1)
            cv2.circle(img_orig, (int((plate[10]) * rx), int((plate[11]) * ry)), 1, (0, 255, 0), -1)

            x1 = int((plate[0] - plate[2] / 2.) * rx)
            y1 = int((plate[1] - plate[3] / 2.) * ry)
            x2 = int((plate[0] + plate[2] / 2.) * rx)
            y2 = int((plate[1] + plate[3] / 2.) * ry)
            cv2.rectangle(img_orig, (x1, y1), (x2, y2), (0, 255, 0), 1)

            plate_box = np.array(
                [(int((plate[4]) * rx), int((plate[5]) * ry)), (int((plate[6]) * rx), int((plate[7]) * ry)),
                 (int((plate[8]) * rx), int((plate[9]) * ry)), (int((plate[10]) * rx), int((plate[11]) * ry))],
                dtype=np.float32)
            RECT_LP_COORS = np.array([[0, 0], [0, plate[3] * ry], [plate[2] * rx, 0], [plate[2] * rx, plate[3] * ry]],
                                     dtype=np.float32)
            transformation_matrix = cv2.getPerspectiveTransform(plate_box, RECT_LP_COORS)
            lp_img = cv2.warpPerspective(img_orig, transformation_matrix,
                                         np.array([plate[2] * rx, plate[3] * ry]).astype(int))
            cv2.imwrite(os.path.join(base_folder, 'logs/exp4/') + os.path.basename(image_path).replace('.' + extension,
                                                                                                       '') + f'_lp_{plate_idx}.jpg',
                        lp_img)
        cv2.imwrite(os.path.join(base_folder, 'logs/exp4/') + os.path.basename(image_path).replace('.' + extension,
                                                                                                   '') + '.jpg',
                    img_orig)
        print(f"Image:{image_path} was processed and written into debug folder")
