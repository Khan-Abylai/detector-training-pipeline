import os
from glob import glob
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import config
import tools.bbox_utils as bu
import onnxruntime as onnxrt

img_w = config.IMG_W
img_h = config.IMG_H
img_size = (img_w, img_h)
DEVICE_NAME = 'cpu'
onnx_session = onnxrt.InferenceSession("/home/user/src/weights/detector_base.onnx", providers=['CPUExecutionProvider'])

ls = glob('/home/user/src/data/single_test_images/*')[:10]

for image_path in ls:
    start_time = time.time()
    original_image = cv2.imread(image_path)

    x = cv2.resize(original_image, (img_w, img_h))
    x = x.transpose((2, 0, 1))
    x = 2 * (x / 255 - 0.5)
    x = torch.FloatTensor(np.array([x.astype(np.float32)])).contiguous()
    binding = onnx_session.io_binding()
    binding.bind_input(name='actual_input', device_type=DEVICE_NAME, device_id=0, element_type=np.float32,
                       shape=tuple(x.shape), buffer_ptr=x.data_ptr(), )

    z_tensor = torch.empty((1, 1024, 13), dtype=torch.float32, device='cpu:0').contiguous()
    binding.bind_output(name='output', device_type=DEVICE_NAME, device_id=0, element_type=np.float32,
                        shape=tuple(z_tensor.shape), buffer_ptr=z_tensor.data_ptr(), )
    onnx_session.run_with_iobinding(binding)
    result = z_tensor.cpu().detach().numpy()
    end_time = time.time() - start_time

    print(f'exec time:{end_time}')

# 0.004422664642333984
# 0.0035326480865478516
# 0.043135881423950195
