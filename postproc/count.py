import numpy as np
import os
import shutil

base_path = '/data/ubuntu/data_3d/output_current_3d_model_sphere'
folder_list = ['tr', 'te', 'te_long', 'tr_long']
for folder in folder_list:
    files = os.listdir(os.path.join(base_path, folder))
    print(f'Folders in {folder} = {len(files)}')
