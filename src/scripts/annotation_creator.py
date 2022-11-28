import os
import numpy as np
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split

folder = '/home/user/mnt/data'

folder_names = ['austria', 'czech', 'finland', 'germany', 'greece', 'italy', 'latvia', 'luxemburg', 'netherlands',
                'romania', 'spain', 'swiss', 'bulgaria', 'estonia', 'france', 'hungary',
                'lithuania', 'montenegro', 'poland', 'portugal', 'slovakia', 'sweden']
# print(len(folder_names))
#
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



for folder_name in folder_names:
    f_name = os.path.join(folder, folder_name, 'filenames.txt')
    df = pd.read_csv(f_name, header=None)
    df.columns = ['filepath']

    train, test = train_test_split(df, test_size=0.05, random_state=42)

    train_path = os.path.join(folder, folder_name, 'train.txt')
    test_path = os.path.join(folder, folder_name, 'test.txt')
    stop = 1
    train.to_csv(train_path, header=None, index_label=False, index=False)
    test.to_csv(test_path, header=None, index_label=False, index=False)


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
