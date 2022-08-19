import os
import numpy as np
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split

folder = '/mnt/data/uk'


folders = glob(os.path.join(folder, "**", "*.jpg"))

data = []


for folder in folders:
    if os.path.exists(folder.replace('.jpg', '.pb')):
        data.append(folder.replace('/mnt/data/uk/', ''))

print(data)
print(len(data))

data = np.array(data)

np.savetxt('/mnt/data/filenames.txt', data, delimiter=" ", fmt="%s")

init_annotation_path = '/mnt/data/filenames.txt'
df = pd.read_csv(init_annotation_path, header=None)
df.columns = ['filepath']

train, test = train_test_split(df, test_size=0.05, random_state=42)

train.to_csv('/mnt/data/train.txt', header=None, index_label=False, index=False)
test.to_csv('/mnt/data/val.txt', header=None, index_label=False, index=False)
#
# df.columns = ['file_path']
# print(df)
#
# df["filepath"] = df['file_path'].apply(lambda x: x.replace('/mnt/8tb', '/mnt/workspace/data'))
#
# df.drop(['file_path'], axis=1, inplace=True)
# df.to_csv('/mnt/workspace/data/filenames.txt', header=None, index_label=False, index=False)
