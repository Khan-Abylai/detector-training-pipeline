# import os
# import cv2
# import numpy as np
# from glob import glob
# import pandas as pd
# from sklearn.model_selection import train_test_split
#
#
# data_folder = '/mnt/data/detector/images'
#
# all_files = glob(os.path.join(data_folder, '*.png'))
#
# print(len(all_files))
#
# filenames = []
#
# for idx, item in enumerate(all_files):
#     if os.path.exists(item.replace(".png", ".pb")):
#         filenames.append(item.replace("/mnt/data/", ""))
#
# filenames = np.array(filenames)
#
# train_filenames, val_filenames = train_test_split(filenames, test_size=0.15, random_state=42)
#
# np.savetxt('/mnt/data/detector/train.txt', train_filenames,  delimiter=" ", fmt="%s")
# np.savetxt('/mnt/data/detector/val.txt', val_filenames,  delimiter=" ", fmt="%s")
#
# folder = '/home/user/mnt/data'
#
# folder_names = ['austria', 'czech', 'finland', 'germany', 'greece', 'italy', 'latvia', 'luxemburg', 'netherlands',
#                 'romania', 'spain', 'swiss', 'bulgaria', 'estonia', 'france',',
#                 'lithuania', 'montenegro', 'poland', 'portugal', 'slovakia', 'sweden']
# print(len(folder_names))
# #
# folder_index = 0
# for folder_name in folder_names:
#     file_index = 0
#     pattern = os.path.join(folder, folder_name, 'images', "**", "*")
#     files = glob(pattern)
#     data = []
#     for file in files:
#         extension = os.path.basename(file).split('.')[-1]
#         if extension == 'pb':
#             continue
#         else:
#             pb_file = os.path.join(os.path.dirname(file), os.path.basename(file).replace(extension, 'pb'))
#             if os.path.exists(pb_file):
#                 appended_data = file.replace(f'{folder}/', '')
#                 data.append(appended_data)
#                 print(f'{appended_data} added. file index:{file_index} folder index:{folder_index}')
#                 file_index += 1
#
#     data = np.array(data)
#
#     folder_index += 1
#     np.savetxt(os.path.join(folder,folder_name, 'filenames.txt'), data, delimiter=" ", fmt="%s")
import os.path

# for folder_name in folder_names:
#     f_name = os.path.join(folder, folder_name, 'filenames.txt')
#     df = pd.read_csv(f_name, header=None)
#     df.columns = ['filepath']
#
#     train, test = train_test_split(df, test_size=0.05, random_state=42)
#
#     train_path = os.path.join(folder, folder_name, 'train.txt')
#     test_path = os.path.join(folder, folder_name, 'test.txt')
#     stop = 1
#     train.to_csv(train_path, header=None, index_label=False, index=False)
#     test.to_csv(test_path, header=None, index_label=False, index=False)


# folder1 = '/mnt/data/uae_data/data'
# folder2 = '/mnt/data/uae_data/new_data'
#
# folder_1_data = glob(os.path.join(folder1, "**", "*"))
# folder_2_data = glob(os.path.join(folder2, "**", "**", "**", "*"))
#
# data_old = []
# data_new = []
#
# for idx, item in enumerate(folder_1_data):
#     extension = os.path.basename(item).split('.')[-1]
#     if extension == 'pb':
#         continue
#     pb_file = os.path.join(os.path.dirname(item), os.path.basename(item).replace(extension, 'pb'))
#
#     if os.path.exists(pb_file):
#         data_old.append(item.replace(os.path.dirname(folder1), '')[1:])
#
#     print(f'working with : {idx} {item}')
#
#
# for idx, item in enumerate(folder_2_data):
#     extension = os.path.basename(item).split('.')[-1]
#     if extension == 'pb':
#         continue
#     pb_file = os.path.join(os.path.dirname(item), os.path.basename(item).replace(extension, 'pb'))
#
#     if os.path.exists(pb_file):
#         data_new.append(item.replace(os.path.dirname(folder2), '')[1:])
#
#     print(f'working with : {idx} {item}')
#
# data = data_old + data_new
#
# print(len(data))
# data = np.array(data)
# np.savetxt('/mnt/data/uae_data/filenames.txt', data, delimiter=" ", fmt="%s")

# folders = glob(os.path.join(folder, "**", "*.jpg"))
#
# data = []
#
#
# for folder in folders:
#     if os.path.exists(folder.replace('.jpg', '.pb')):
#         data.append(folder.replace('/mnt/data/uk/', ''))
#
# print(data)
# print(len(data))
#
# data = np.array(data)
#
# np.savetxt('/mnt/data/filenames.txt', data, delimiter=" ", fmt="%s")
#
# init_annotation_path = '/home/user/mnt/data/uae/images/filenames.txt'
# df = pd.read_csv(init_annotation_path, header=None)
# df.columns = ['filepath']
#
# train, test = train_test_split(df, test_size=0.05, random_state=42)
#
# train.to_csv('/home/user/mnt/data/uae/images/train.txt', header=None, index_label=False, index=False)
# test.to_csv('/home/user/mnt/data/uae/images/val.txt', header=None, index_label=False, index=False)

# df.columns = ['file_path']
# print(df)
#
# df["filepath"] = df['file_path'].apply(lambda x: x.replace('/mnt/8tb', '/mnt/workspace/data'))
#
# df.drop(['file_path'], axis=1, inplace=True)
# df.to_csv('/mnt/workspace/data/filenames.txt', header=None, index_label=False, index=False)

# base_folder = '/home/user/mnt/data/uae/images/**/**/*'
#
# images = np.array([x.replace('/home/user/mnt/data/', '') for x in glob(base_folder) if '.jpg' in x or '.jpeg' in x or '.png' in x])
# df = pd.DataFrame(data = images)
# df.to_csv('/home/user/mnt/data/uae/images/filenames.txt', header=None, index_label=False, index=False)
#
# stop = 1

# df = pd.read_csv("/mnt/kz_data/kz_new/train_kz.txt")
# df.columns = ["filename"]
# df['local_fname'] = df['filename'].apply(lambda x:x.replace("/mnt/data/", "/mnt/kz_data/"))
# df.drop(["filename"], axis=1, inplace=True)
# df.to_csv("/mnt/kz_data/kz_new/train_detector_kz.txt", header=None, index_label=False, index=False)


# from glob import glob
# import os
#
#
# def draw_annotation(img_orig, plate):
#     cv2.circle(img_orig, (int(plate[0]), int(plate[1])), 1, (0, 255, 255), -1)
#
#     cv2.circle(img_orig, (int((plate[4])), int((plate[5]))), 2, (0, 255, 0), -1)
#     cv2.circle(img_orig, (int((plate[6])), int((plate[7]))), 3, (0, 255, 0), -1)
#     cv2.circle(img_orig, (int((plate[8])), int((plate[9]))), 4, (0, 255, 0), -1)
#     cv2.circle(img_orig, (int((plate[10])), int((plate[11]))), 5, (0, 255, 0), -1)
#
#     # x1 = int((plate[0] - plate[2] / 2.))
#     # y1 = int((plate[1] - plate[3] / 2.))
#     # x2 = int((plate[0] + plate[2] / 2.))
#     # y2 = int((plate[1] + plate[3] / 2.))
#     # cv2.rectangle(img_orig, (x1, y1), (x2, y2), (0, 255, 0), 1)
#     return img_orig
#
#
# def convert_yolo_annotation(arr):
#     """Converts a YOLO-type annotation (center_x, center_y, w, h) to the specified format.
#
#     Args:
#         center_x: The x-coordinate of the object's center.
#         center_y: The y-coordinate of the object's center.
#         w: The width of the object.
#         h: The height of the object.
#
#     Returns:
#         A list containing the annotation in the specified format:
#         [center_x, center_y, h, w, top_left_x, top_left_y, bottom_left_x, bottom_left_y,
#          top_right_x, top_right_y, bottom_right_x, bottom_right_y]
#     """
#     center_x, center_y, w, h = arr
#     # Calculate the top-left corner coordinates
#     top_left_x = int(center_x - w / 2)
#     top_left_y = int(center_y - h / 2)
#
#     # Calculate the other corner coordinates
#     bottom_left_x = top_left_x
#     bottom_left_y = top_left_y + h
#     top_right_x = top_left_x + w
#     top_right_y = top_left_y
#     bottom_right_x = top_right_x
#     bottom_right_y = bottom_left_y
#
#     # Arrange the values in the specified order
#     return [center_x, center_y, w, h, top_left_x, top_left_y, bottom_left_x, bottom_left_y, top_right_x, top_right_y,
#             bottom_right_x, bottom_right_y]
#
#
# def draw_rectangle(image, arr, color=(0, 255, 0), thickness=2):
#     x, y, w, h = arr
#     # Calculate the top-left corner coordinates from the center coordinates
#     start_x = int(x - w / 2)
#     start_y = int(y - h / 2)
#
#     # Draw the rectangle using the calculated top-left corner and width/height
#     cv2.rectangle(image, (start_x, start_y), (start_x + w, start_y + h), color, thickness)
#     return image
#
#
# data_folder = '/mnt/data/detector/images'
# label_folder = '/mnt/data/detector/labels'
# all_images = glob(os.path.join(data_folder, '*.png'))
# all_annotations = glob(os.path.join(label_folder, "*"))
#
# for idx, image_path in enumerate(all_images):
#
#     print(idx, image_path)
#     label_path = image_path.replace("/images/", "/labels/").replace(".png", ".txt")
#     image = cv2.imread(image_path)
#     if image is None:
#         continue
#     h, w, _ = image.shape
#     if label_path in all_annotations:
#         with open(label_path, "r") as f:
#             content = np.array(
#                 [[float(y) for y in x.replace("\t", "").replace("\n", "").split()] for x in f.readlines()])
#
#         if content.shape[0] != 0:
#
#             print(content)
#             content = content[:, 1:]
#             content[:, ::2] *= w
#             content[:, 1::2] *= h
#
#             full_annotation = np.array([convert_yolo_annotation(x) for x in content]).reshape(-1, 12)
#             full_annotation[:, ::2] /= w
#             full_annotation[:, 1::2] /= h
#             pb_file_path = image_path.replace(".png", ".pb")
#             np.savetxt(pb_file_path, full_annotation)

#
# import os
# import shutil
# from glob import glob
#
# import numpy as np

# anns = []
# prefix = '/mnt/data/recognizer/'
# base_folder = '/mnt/data/recognizer/jan-2024-iteration/lp_images'
# images = glob(os.path.join(base_folder, "*/*/*/*/*.jpeg")) + glob(os.path.join(base_folder, "*/*/*/*/*/*.jpeg"))
# annotation_path = '/mnt/data/recognizer/jan-2024-iteration/jan-2024-iteration.csv'
# anns.append(["image_path,car_labels"])
# for idx,image in enumerate(images):
#     approx_path = image.replace("/lp_images/", "/for_annotation/").replace(".jpeg", ".txt").replace("_license_plate.",".")
#     print(idx,approx_path)
#     if os.path.exists(image) and os.path.exists(approx_path):
#         new_path = approx_path.replace("/for_annotation/", "/lp_images/")
#         shutil.copy(approx_path, new_path)
#         image = image.replace(prefix, '')
#
#         with open(approx_path, "r") as f:
#             content=f.read().strip()
#         ann = ",".join([os.path.join("/data", image), content])
#         anns.append(ann)
#         stop = 1
# anns = np.array(anns)
# np.savetxt(annotation_path, anns, delimiter=" ", fmt="%s")
# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# df1 = pd.read_csv("/mnt/data/USA_RELEASE_2/detector/all_files.txt")
# df1.columns = ['filename']
# df2 = pd.read_csv("/mnt/data/USA_RELEASE_2/detector/jan-2024-filenames.txt")
# df2.columns = ['filename']
# df = pd.concat([df1, df2], axis=0, ignore_index=True)
#
# train, test = train_test_split(df, test_size=0.1, random_state=42)
#
#
# train.to_csv('/mnt/data/USA_RELEASE_2/detector/train.txt', index=False, index_label=False)
# test.to_csv('/mnt/data/USA_RELEASE_2/detector/test.txt', index=False, index_label=False)
#
# stop = 1

import numpy as np
import cv2
import pandas as pd
from glob import glob

from tqdm import tqdm

content = glob("../../data/zoning/*/*")

for item in tqdm(content):
    basename = os.path.basename(item)
    filename, ext = os.path.splitext(basename)

    if ext == '.pb':
        continue
    if not os.path.exists(item.replace(ext, ".pb")):
        continue
    pb_file_path = item.replace(ext, ".pb")
    pb_content = np.loadtxt(pb_file_path).reshape(-1, 12)

    image = cv2.imread(item)
    h_, w_, _ = image.shape
    pb_content[:, ::2] *= w_
    pb_content[:, 1::2] *= h_

    plate = pb_content[0].reshape(-1, 2).astype(int)
    cp, size, lt, lb, rt, rb = plate
    w,h = size
    coords = np.array([
        [0,0], [0, h-1],[w-1, 0],[w-1, h-1]
    ], dtype='float32')

    bbox = np.array([lt, lb, rt, rb], dtype='float32')
    sizes = size.astype(int)
    transformation_matrix = cv2.getPerspectiveTransform(bbox, coords)
    lp_img = cv2.warpPerspective(image, transformation_matrix, sizes)

    cv2.imwrite(f'../../logs/exp2/{filename}_lp.jpg', lp_img)
