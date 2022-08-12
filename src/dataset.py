from torch.utils.data import Dataset
import numpy as np
import cv2
import warnings
import os
import pandas as pd
import lmdb
import pyarrow as pa
import six
from tools import transforms
import config as config
from PIL import Image
import uuid

count = 0


class LPDataset(Dataset):
    def __init__(self, txt_files, transforms, size=(512, 512), data_dir='', train=False, debug=False,
                 return_filepath=False):
        image_filenames = []
        for txt_file in txt_files:
            with open(os.path.join(data_dir, txt_file)) as f:
                image_filenames.append(np.array(f.read().splitlines()))
        self.image_filenames = np.concatenate(image_filenames, axis=0)
        if train:
            np.random.shuffle(self.image_filenames)

        self.size = size
        self.transformation = transforms
        self.data_dir = data_dir
        self.debug = debug
        self.return_filepath = return_filepath
        stop = 1

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_filename = os.path.join(self.data_dir, self.image_filenames[index])
        if not os.path.exists(image_filename):
            print("no file")
            return self[(index + 1) % len(self)]
        with open(image_filename, 'rb') as f:
            check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            return self[(index + 1) % len(self)]

        plate_filename = image_filename.replace('.jpg', '.pb').replace('.jpeg', '.pb').replace('.png', '.pb')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            image = cv2.imread(image_filename)
            img_h, img_w, img_c = image.shape

            plate_boxes = np.loadtxt(plate_filename).reshape(-1, 12)
            checker_mask_1 = plate_boxes > 1.0
            checker_mask_2 = plate_boxes <= 0.0

            if True in checker_mask_2:
                return self[(index + 1) % len(self)]

            if np.all(checker_mask_1):
                pass
            else:
                plate_boxes[:, ::2] *= img_w
                plate_boxes[:, 1::2] *= img_h

            if self.debug:
                copy_img = image.copy()
                img_path = os.path.basename(image_filename).replace('.', '_resized_1.')
                for plate in plate_boxes:
                    cv2.circle(copy_img, plate[0:2].astype(int), radius=1, color=(0, 0, 255), thickness=-1)
                    cv2.circle(copy_img, plate[4:6].astype(int), radius=1, color=(0, 255, 255), thickness=-1)
                    cv2.circle(copy_img, plate[4:6].astype(int), radius=1, color=(0, 255, 255), thickness=-1)
                    cv2.circle(copy_img, plate[6:8].astype(int), radius=1, color=(0, 255, 255), thickness=-1)
                    cv2.circle(copy_img, plate[8:10].astype(int), radius=1, color=(0, 255, 255), thickness=-1)
                    cv2.circle(copy_img, plate[10:12].astype(int), radius=1, color=(0, 255, 255), thickness=-1)
                cv2.imwrite(os.path.join(config.DEBUG_FOLDER, 'exp1', img_path), copy_img)

            plate_boxes[:, ::2] *= config.IMG_W / img_w
            plate_boxes[:, 1::2] *= config.IMG_H / img_h
            if self.debug:
                copy_img = cv2.resize(image, self.size)
                for plate in plate_boxes:
                    cv2.circle(copy_img, plate[0:2].astype(int), radius=1, color=(0, 0, 255), thickness=-1)
                    cv2.circle(copy_img, plate[4:6].astype(int), radius=1, color=(0, 255, 255), thickness=-1)
                    cv2.circle(copy_img, plate[4:6].astype(int), radius=1, color=(0, 255, 255), thickness=-1)
                    cv2.circle(copy_img, plate[6:8].astype(int), radius=1, color=(0, 255, 255), thickness=-1)
                    cv2.circle(copy_img, plate[8:10].astype(int), radius=1, color=(0, 255, 255), thickness=-1)
                    cv2.circle(copy_img, plate[10:12].astype(int), radius=1, color=(0, 255, 255), thickness=-1)
                img_path = os.path.basename(image_filename).replace('.', '_resized_2.')
                cv2.imwrite(os.path.join(config.DEBUG_FOLDER, 'exp1', img_path), copy_img)

            image = cv2.resize(image, self.size)
            plate_boxes[:, ::2] /= config.IMG_W
            plate_boxes[:, 1::2] /= config.IMG_H

            plate_boxes[:, [4, 6, 8, 10]] -= plate_boxes[:, [0]]
            plate_boxes[:, [5, 7, 9, 11]] -= plate_boxes[:, [1]]
            stop = 1
            if self.return_filepath:
                return self.transformation(image, plate_boxes), image_filename
            else:
                return self.transformation(image, plate_boxes)


if __name__ == '__main__':
    train_transforms = transforms.DualCompose([transforms.OneOf(
        [transforms.ImageOnly(transforms.GaussianBlur()), transforms.ImageOnly(transforms.AverageBlur()), ], prob=0.2),
        transforms.ImageOnly(transforms.RandomChannelPermute(prob=0.2)),  #
        transforms.OneOf(
            [transforms.ImageOnly(transforms.ContrastNormalization()), transforms.ImageOnly(transforms.Add()), ],
            prob=0.2), transforms.ImageOnly(transforms.AdditiveGaussianNoise(prob=0.2)), transforms.OneOf(
            [transforms.ImageOnly(transforms.SaltAndPepper()), transforms.ImageOnly(transforms.Dropout()), ], prob=0.2),
        transforms.ImageOnly(transforms.RandomGrayScale(prob=0.2)), transforms.HorizontalFlip(prob=0.2),
        transforms.Rotate(limit=15, prob=0.2), transforms.ScaleDown(scale=0.5, prob=0.5),
        transforms.ImageOnly(transforms.Transpose()), transforms.Normalize(), transforms.BoxOnly(transforms.FillBox()),
        transforms.ToTensor()])

    visible_transform = transforms.DualCompose([transforms.OneOf(
        [transforms.ImageOnly(transforms.GaussianBlur()), transforms.ImageOnly(transforms.AverageBlur()), ], prob=0.2),
        transforms.ImageOnly(transforms.RandomChannelPermute(prob=0.2)),  #
        transforms.OneOf(
            [transforms.ImageOnly(transforms.ContrastNormalization()), transforms.ImageOnly(transforms.Add()), ],
            prob=0.2), transforms.ImageOnly(transforms.AdditiveGaussianNoise(prob=0.2)), transforms.OneOf(
            [transforms.ImageOnly(transforms.SaltAndPepper()), transforms.ImageOnly(transforms.Dropout()), ], prob=0.2),
        transforms.ImageOnly(transforms.RandomGrayScale(prob=0.2)), transforms.HorizontalFlip(prob=0.2),
        transforms.Rotate(limit=15, prob=0.2), transforms.ScaleDown(scale=0.5, prob=0.5),
        transforms.ImageOnly(transforms.Transpose()), transforms.BoxOnly(transforms.FillBox())])

    lp_dataset = LPDataset(['/mnt/workspace/uae_data/train.txt'], transforms=visible_transform, size=(512, 512),
                           data_dir='/mnt/workspace/uae_data', train=True, debug=True, return_filepath=True)
    for idx, item in enumerate(lp_dataset):
        image_bboxes, filename = item
        image, bboxes = image_bboxes
        bboxes[:, [4, 6, 8, 10]] += bboxes[:, [0]]
        bboxes[:, [5, 7, 9, 11]] += bboxes[:, [1]]

        bboxes[:, ::2] *= 512
        bboxes[:, 1::2] *= 512
        image = image.transpose(1, 2, 0)
        image = np.ascontiguousarray(image, dtype=np.uint8)
        for plate in bboxes:
            cv2.circle(image, plate[0:2].astype(int), radius=1, color=(0, 0, 255), thickness=-1)
            cv2.circle(image, plate[4:6].astype(int), radius=1, color=(0, 255, 255), thickness=-1)
            cv2.circle(image, plate[4:6].astype(int), radius=1, color=(0, 255, 255), thickness=-1)
            cv2.circle(image, plate[6:8].astype(int), radius=1, color=(0, 255, 255), thickness=-1)
            cv2.circle(image, plate[8:10].astype(int), radius=1, color=(0, 255, 255), thickness=-1)
            cv2.circle(image, plate[10:12].astype(int), radius=1, color=(0, 255, 255), thickness=-1)
            path = os.path.join(config.DEBUG_FOLDER, 'exp1', os.path.basename(filename).replace('.', '_resized_3.'))
            print(f"Path:{path} was written")
            print(image.shape)
            cv2.imwrite(path, image)
