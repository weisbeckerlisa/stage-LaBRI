from skimage.transform import resize
from skimage.util import img_as_float
from skimage.segmentation import slic
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import kornia
import torch
import shutil
import os
from skimage import io
from skimage.color import rgb2gray, rgb2lab
#from inference import inference


def save_checkpoint(state, is_best, checkpoint_dir, idx, best_model_dir):
    f_path = checkpoint_dir + '/checkpoint_'+str(idx)+'.pt'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, f_path)
    if is_best:
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        best_fpath = best_model_dir + '/best_model_'+str(idx)+'.pt'
        shutil.copyfile(f_path, best_fpath)


def load_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def img_segments_only(img_grays, div_resize, num_max_seg):
    """
    :param img_grays: Target image
    :param div_resize: Resizing factor
    :return: Superpixel's label map for the target and reference images.
    """

    # Gray scale or RGB image
    if img_grays.ndim == 2:
        image_gray_2 = img_as_float(resize(
            img_grays, (img_grays[0].size / div_resize, img_grays[1].size / div_resize)))
        segment_gray_2 = slic(image_gray_2, n_segments=int(
            num_max_seg), sigma=0, compactness=0.1, channel_axis=None)
        #use weight from ./log/bset_model.pth
        #segment_gray_2 = inference(image_gray_2, nspix=int(num_max_seg), n_iter=10, fdim=20, weight='./log/bset_model.pth', color_scale=0.9, pos_scale=2.0)

    else:
        image_gray_2 = img_as_float(resize(
            img_grays, (len(img_grays[0]) / div_resize, len(img_grays[1]) / div_resize)))
        segment_gray_2 = slic(image_gray_2, n_segments=int(
            num_max_seg), sigma=0, compactness=9)
        #segment_gray_2 = inference(image_gray_2, nspix=int(num_max_seg), n_iter=10, weight='./log/bset_model.pth', color_scale=0.9, pos_scale=2.0)

    return segment_gray_2


def imagenet_norm(input, device):
    # VGG19 normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).to(
        device=device, dtype=torch.float)
    std = torch.tensor([0.229, 0.224, 0.225]).to(
        device=device, dtype=torch.float)
    out = (input - mean.view((1, 3, 1, 1))) / std.view((1, 3, 1, 1))
    return out

# : COMMENT : ANGELIKA : 09/03/2023 : sauvegarde la carte d'attention en format image et npy
def save_attention_maps(superattention_maps, path_map):
    """
    Get the figure of the 'superattention_map' as an image and a npy file.
    """
    idx = 1
    for att_map in superattention_maps:
        
        att_map = att_map.unsqueeze(0)
        #att_map = kornia.utils.tensor_to_image(att_map[0,0:1, : ,:].cpu())
        att_map = kornia.utils.tensor_to_image(att_map[0:1, : ,:].cpu())


        ax = plt.subplot()
        im = ax.imshow(att_map)

        # create an Axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        plt.savefig(path_map + 'superattention_' + str(idx) + '.png')
        np.save(path_map +'superattention_' + str(idx) + '.npy', att_map)
        plt.close(fig=None)
        
        idx+=1
    print("attention maps have been saved in the directory \'" + path_map + "\'\n")
        

def get_corresponding_superpixel(attention_map, target_spixel_index):
    """
    Given an attention map and the index of a superpixel in the target image, 
    return the index of the corresponding superpixel in the reference image.

    :param attention_map: 2D matrix representing the attention map.
    :param target_spixel_index: The index of the superpixel in the target image.
    :return: The index of the corresponding superpixel in the reference image.
    """
    target_spixel_attention = attention_map[target_spixel_index, :]  # Retrieve the attention scores for the target superpixel
    corresponding_spixel_index = np.argmax(target_spixel_attention)  # Find the index of the max attention score
    return corresponding_spixel_index


def get_high_attention_values(attention_map, target_label):

    # Extract the corresponding row from the attention map
    attention_row = attention_map[target_label, :]


    # Threshold for attention scores
    threshold = min(attention_row.max(), 0.1)

    # Find labels with high attention
    high_attention_indices = np.where(attention_row > threshold)

    # Return labels and their corresponding attention scores
    return high_attention_indices, attention_row[high_attention_indices]

def preprocess(target_path, ref_path, target_result_path, ref_result_path):
    ######## accv_img #############

        size = 224

        nb_img = len(os.listdir(target_path))

        for i in range (nb_img):
            target_real = rgb2gray(io.imread(target_path + str(i) + '_target.png', pilmode='RGB'))
            # Reading target images in RGB
            target = rgb2gray(resize(io.imread(target_path + str(i) + '_target.png', pilmode='RGB'), (224, 224)))
            ref_real = io.imread(ref_path +  str(i) + '_ref.png', pilmode='RGB')
            # Reading ref images in RGB
            ref = resize(io.imread(ref_path + str(i) + '_ref.png', pilmode='RGB'), (224, 224))
            if np.ndim(target) == 3:
                target_luminance_classic = (target[:, :, 0])

            else:
                target_luminance_classic = (target)
                target = target[:, :, np.newaxis]
                target_real = target_real[:, :, np.newaxis]

            ref_new_color = rgb2lab(ref)
            ref_luminance_classic = (ref_new_color[:, :, 0] / 100.0)

            ref_luminance = ref_luminance_classic

            # Save target luminance_classic, ref_luminance_classic pour
            # traitement sur Matlab

            for resize_factor in [1, 2, 4, 8]:
                target_resize = resize(target_luminance_classic, (size / resize_factor, size / resize_factor))
                ref_resize = resize(ref_luminance, (size / resize_factor, size / resize_factor))
                plt.imsave(target_result_path + str(i)+ '_target_' + str(resize_factor) + '.png', target_resize)
                plt.imsave(ref_result_path + str(i)+ '_ref_' + str(resize_factor) + '.png', ref_resize)
