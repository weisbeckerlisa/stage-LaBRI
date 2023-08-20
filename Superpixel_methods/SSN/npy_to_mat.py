import numpy as np
from scipy.io import savemat
import os

npy_dir = './res_deep/'  # Input directory
output_dir = './res_deep_mat/'  # Output directory

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file in os.listdir(npy_dir):
    if file.endswith('.npy'):
        npy_file = os.path.join(npy_dir, file)
        label = np.load(npy_file)

        mat_dict = {}
        mat_dict['label'] = label

        # Save as .mat file
        mat_file = os.path.join(output_dir, file.replace('.npy', '.mat'))
        savemat(mat_file, mat_dict)
