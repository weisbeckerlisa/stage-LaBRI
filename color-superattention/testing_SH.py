import torch
from torchvision.utils import save_image
from utils import imagenet_norm, save_attention_maps
import os


def testing_color_SH(loader_val, model_color, device, mode, save_maps):

    # Save path for resulting images
    path = './results/super_attent_v1_res_conection/unicolor_img/epoch35/'

    if not os.path.exists(path):
        os.makedirs(path)

    with torch.no_grad():

        model_color.eval()
        for idx, (img_rgb_target, img_target_gray, ref_rgb, ref_gray,
                    target_slic, ref_slic_all, img_ref_ab,
                    img_gray_map, gray_real, ref_real) in enumerate(loader_val):
            if idx % 1 == 0:
                # Target data
                img_gray_map = (img_gray_map).to(
                    device=device, dtype=torch.float)
                img_target_gray = (img_target_gray).to(
                    device=device, dtype=torch.float)
                gray_real = gray_real.to(device=device, dtype=torch.float)
                target_slic = target_slic

                # Loading references
                ref_rgb_torch = ref_rgb.to(
                    device=device, dtype=torch.float)
                img_ref_gray = (ref_gray).to(
                    device=device, dtype=torch.float)
                ref_slic_all = ref_slic_all

                # VGG19 normalization
                img_ref_rgb_norm = imagenet_norm(ref_rgb_torch, device)

                # VGG19 normalization
                img_target_gray_norm = img_target_gray
                img_ref_gray_norm = img_ref_gray

                _, _, pred_rgb_torch, attention_maps = model_color(img_target_gray_norm,
                                                    img_ref_gray_norm,
                                                    img_target_gray,
                                                    target_slic,
                                                    ref_slic_all,
                                                    img_gray_map,
                                                    gray_real,
                                                    img_ref_rgb_norm,
                                                    device, mode)                 
                save_image(pred_rgb_torch,
                            path + str(idx) + '_pred.png',
                            normalize=True)

                print(path + str(idx) + '_pred.png')

                print('save image')

                if save_maps == True:
                    path_attention_map = path + 'attention_maps/' + str(idx) + '/'
                    if not os.path.exists(path_attention_map):
                        os.makedirs(path_attention_map)

                    save_attention_maps(attention_maps, path_attention_map)
