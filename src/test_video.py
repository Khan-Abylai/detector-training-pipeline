import cv2
import numpy as np
import torch
import torch.nn as nn

import tools.bbox_utils as bu
from models.models import LPDetector
from tools import transforms

img_w = 512
img_h = 512
img_size = (img_w, img_h)
model = LPDetector(img_size).cuda()
checkpoint = '/mnt/workspace/model_dir_lp_detector/499_500_TRAIN_|_Plates_0.000_659__Loss_0.083_VAL_|_Plates_Recall_0.969_124_Val_loss_0.055,_lr=1.220703125e-06.pth'
model = nn.DataParallel(model)
checkpoint = torch.load(checkpoint)['state_dict']
model.load_state_dict(checkpoint)
model.eval()

transform = transforms.DualCompose([
    transforms.ImageOnly(transforms.Transpose()),
    transforms.Normalize(),
    transforms.ToTensor()
])

cap = cv2.VideoCapture('/mnt/workspace/experiments/video/video_test.mp4')

index = 0
while True:

    ret, img = cap.read()

    if ret:
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

                plate_box = np.array([
                    (int((plate[4]) * rx), int((plate[5]) * ry)),
                    (int((plate[6]) * rx), int((plate[7]) * ry)),
                    (int((plate[8]) * rx), int((plate[9]) * ry)),
                    (int((plate[10]) * rx), int((plate[11]) * ry))], dtype=np.float32)
                RECT_LP_COORS = np.array([
                    [0, 0],
                    [0, plate[3] * ry],
                    [plate[2] * rx, 0],
                    [plate[2] * rx, plate[3] * ry]], dtype=np.float32)
                transformation_matrix = cv2.getPerspectiveTransform(plate_box, RECT_LP_COORS)
                lp_img = cv2.warpPerspective(img_orig, transformation_matrix,
                                             np.array([plate[2] * rx, plate[3] * ry]).astype(int))
                cv2.imwrite('/mnt/workspace/experiments/debug/' + str(index) + f'_lp_{plate_idx}.jpg', lp_img)
            cv2.imwrite('/mnt/workspace/experiments/debug/' + str(index) + '.jpg', img_orig)
            print(f"Image with idx:{index} was processed and written")
        index += 1
    else:
        break
