import numpy as np
import os
import shutil

files = os.listdir()
long_files = []
print(len(files))

for file in files:
    if len(file) == 7:
        long_files.append(file)

print(len(long_files))

base_tr = '/data/ubuntu/data_3d/output_current_3d_model_sphere/te'
long_tr = '/data/ubuntu/data_3d/output_current_3d_model_sphere/te_long'

for file in long_files:
    old_path = os.path.join(base_tr, file)
    new_path = os.path.join(long_tr, file)

    print('old path', old_path)
    print('new_path', new_path)

    shutil.move(old_path, new_path)

