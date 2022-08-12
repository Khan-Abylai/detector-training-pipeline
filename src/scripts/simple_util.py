import os
from glob import glob
import numpy as np
data_folder = '/mnt/workspace/uae_data'

old_data = [x.replace(data_folder+'/', '') for x in glob(os.path.join(data_folder, 'data', '**', "*"))if '.pb' not in x and '.txt' not in x ]
new_data = [x.replace(data_folder+'/', '') for x in glob(os.path.join(data_folder, 'new_data', '**', "**", "**", "*")) if '.pb' not in x and '.txt' not in x]

new_old_data = []
new_new_data = []

for idx,item in enumerate(old_data):
    extension = os.path.basename(item).split('.')[-1]
    pb_file = os.path.join(os.path.dirname(item), os.path.basename(item).replace(extension, 'pb'))
    if os.path.exists(os.path.join(data_folder, pb_file)):
        plate_boxes = np.loadtxt(os.path.join(data_folder, pb_file)).reshape(-1, 12)
        checker_mask_2 = plate_boxes <= 0.0
        if True in checker_mask_2:
            print(
                '--------------------------------------------------------------------------------------------------------')
            continue
        new_old_data.append(item)
        print(idx)

for idx,item in enumerate(new_data):
    extension = os.path.basename(item).split('.')[-1]
    pb_file = os.path.join(os.path.dirname(item), os.path.basename(item).replace(extension, 'pb'))
    if os.path.exists(os.path.join(data_folder, pb_file)):
        plate_boxes = np.loadtxt(os.path.join(data_folder, pb_file)).reshape(-1, 12)
        checker_mask_2 = plate_boxes <= 0.0
        if True in checker_mask_2:
            print('--------------------------------------------------------------------------------------------------------')
            continue
        new_new_data.append(item)
        print(idx)


data = new_old_data + new_new_data

data = np.array(data)
np.savetxt('/mnt/workspace/uae_data/filenames.txt', data, delimiter=" ", fmt="%s")
