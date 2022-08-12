import os
import numpy as np
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split

init_annotation_path = '/mnt/workspace/uae_data/filenames.txt'
df = pd.read_csv(init_annotation_path, header=None)
df.columns = ['filepath']

train, test = train_test_split(df, test_size=0.05, random_state=42)

train.to_csv('/mnt/workspace/uae_data/train.txt', header=None, index_label=False, index=False)
test.to_csv('/mnt/workspace/uae_data/val.txt', header=None, index_label=False, index=False)

# df.columns = ['file_path']
# print(df)
#
# df["filepath"] = df['file_path'].apply(lambda x: x.replace('/mnt/8tb', '/mnt/workspace/data'))
#
# df.drop(['file_path'], axis=1, inplace=True)
# df.to_csv('/mnt/workspace/data/filenames.txt', header=None, index_label=False, index=False)
