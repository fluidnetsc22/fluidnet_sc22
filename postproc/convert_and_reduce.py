import numpy as np
import torch
import os

base_path = '/data/ubuntu/data_3d/output_current_3d_model_sphere'
new_path = '/data/ubuntu/data_3d_reduced/output_current_3d_model_sphere'
folder_list = ['tr', 'te']
n_folders = 320
for folder in folder_list:
    for i in range(n_folders):

        # If does not exist, create a folder in source
        if not os.path.exists(os.path.join(new_path, folder, '{:06}'.format(i))):
            os.makedirs(os.path.join(new_path, folder, '{:06}'.format(i)))

        # Sweep the 64 files in each folder
        for file in os.listdir(os.path.join(base_path, folder, '{:06}'.format(i))):
            if file.endswith(".pt"):
                print('File name', os.path.join(os.path.join(base_path, folder, '{:06}'.format(i), file)))
                loaded = torch.load(os.path.join(os.path.join(base_path, folder, '{:06}'.format(i), file)))
                new_file = loaded[:, :, 16:48, 16:48,16:48]
                print('New name', os.path.join(os.path.join(new_path, folder, '{:06}'.format(i), file)))
                torch.save(new_file, os.path.join(os.path.join(new_path, folder, '{:06}'.format(i), file)))

