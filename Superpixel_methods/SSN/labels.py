import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import savemat

for segmentations in os.listdir("./cocoval_gt"):

    seg = plt.imread("./cocoval_gt/" + segmentations)


    if seg.shape[0] == 1:
        img = np.repeat(img, 3, axis = 0)

    seg_img_encoded = seg[..., 0] * 256 * 256 + seg[..., 1] * 256 + seg[..., 2]
    labels = np.zeros(seg_img_encoded.shape, dtype=np.int32)

    unique_classes = np.unique(seg_img_encoded)
    for i, unique_class in enumerate(unique_classes):
        if i >= 50: break
        labels[seg_img_encoded == unique_class] = i

    # save labels as .mat
    dest_gt = './cocoval_mat/'+segmentations[0:-4]+'.mat'
    savemat(dest_gt, {'label': labels})
