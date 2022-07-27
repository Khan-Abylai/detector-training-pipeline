import torch.nn as nn
import torch
from src.models.blocks import LinearConvBlock, OrdinaryConvBlock, PlateYoloBlock
import src.config as config


class LPDetector(nn.Module):

    def __init__(self, image_size):
        super().__init__()

        self.plate_features = nn.Sequential(OrdinaryConvBlock(3, 16), OrdinaryConvBlock(16, 16),
                                            nn.MaxPool2d(kernel_size=2, stride=2), OrdinaryConvBlock(16, 32),
                                            OrdinaryConvBlock(32, 32), nn.MaxPool2d(kernel_size=2, stride=2),
                                            OrdinaryConvBlock(32, 64), OrdinaryConvBlock(64, 64),
                                            nn.MaxPool2d(kernel_size=2, stride=2), OrdinaryConvBlock(64, 128),
                                            OrdinaryConvBlock(128, 128), nn.MaxPool2d(kernel_size=2, stride=2),
                                            OrdinaryConvBlock(128, 256), OrdinaryConvBlock(256, 256), )
        self.yolo_plates = nn.ModuleList()
        self.yolo_plates.append(
            LinearConvBlock(in_channels=256, out_channels=config.PLATE_COORDINATE_DIMENSIONS, kernel_size=1, stride=1))
        self.yolo_plates.append(
            PlateYoloBlock(image_size, grid_size=16, iou_threshold=0.7, conf_threshold=0.8, conf_scale=2, coord_scale=2,
                           noobject_scale=0.08, ))

    def forward(self, imgs, plate_boxes=None, validate=False):
        yolo_plates_output = self.plate_features(imgs)

        for module in self.yolo_plates:
            yolo_plates_output = module(yolo_plates_output, plate_boxes, validate=validate)

        return yolo_plates_output
