import torch
from skimage import color, io, transform
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from utils import img_segments_only, os
from skimage.color import rgb2gray
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from scipy.io import loadmat


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        img = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)

        return img


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img


class MyDataTrain(Dataset):
    def __init__(self, target_path, ref_sim, slic_target, transform=None,
                 target_transfom=None, slic=True, size=224, color_space=None):
        self.target_path = target_path
        self.slic = slic
        self.transform = transform
        self.target_transform = target_transfom
        self.size = size
        self.color_space = color_space
        self.slic_target = slic_target
        self.ref_sim = ref_sim

    def __getitem__(self, index):

        # Randon variable for choosing reference
        r = np.random.randint(1, 3)

        # Loading target and reference images
        target_real = io.imread('/data2/hcarrillolin/PhD/dataset/img/'
                                + str(self.ref_sim[index][0])
                                + '/' + str(self.ref_sim[index][1])
                                + '.JPEG', pilmode='RGB')
        # Reading target images in RGB
        target = resize(io.imread('/data2/hcarrillolin/PhD/dataset/img/'
                                  + str(self.ref_sim[index][0])
                                  + '/' + str(self.ref_sim[index][1])
                                  + '.JPEG'), (224, 224))

        ref_real = io.imread('/data2/hcarrillolin/PhD/dataset/img/'
                             + str(self.ref_sim[index][0])
                             + '/' + str(self.ref_sim[index][r])
                             + '.JPEG', pilmode='RGB')
        # Reading ref images in RGB
        ref = resize(io.imread('/data2/hcarrillolin/PhD/dataset/img/'
                               + str(self.ref_sim[index][0])
                               + '/' + str(self.ref_sim[index][r])
                               + '.JPEG', pilmode='RGB'), (224, 224))

        # Passing RGB to LAB color space
        if self.color_space == 'lab':
            target_new_color = color.rgb2lab(target)
            target_real_lab = color.rgb2lab(target_real)
            target_luminance_classic_real = (target_real_lab[:, :, 0] / 100.0)
            target_luminance_classic = (target_new_color[:, :, 0] / 100.0)

            target_chroma = target_new_color[:, :, 1:] / 127.0  # [-1, 1]

            ref_new_color = color.rgb2lab(ref)
            ref_luminance_classic = (ref_new_color[:, :, 0] / 100.0)

            ref_chroma = ref_new_color[:, :, 1:] / 127.0

            # Luminance remapping
            target_luminance_map = (
                (np.std(ref_luminance_classic) / np.std(target_luminance_classic))
                * (target_luminance_classic - np.mean(target_luminance_classic))
                + np.mean(ref_luminance_classic))

            ref_luminance = ref_luminance_classic


        # Calculating superpixel label map 
        # for target and reference images (Grayscale)

        target_slic = img_segments_only(target_luminance_classic, 1, self.size)
        ref_slic = img_segments_only(ref_luminance, 1, self.size)

        target_slic_2 = img_segments_only(
            target_luminance_classic, 2, int(self.size / 2))

        ref_slic_2 = img_segments_only(ref_luminance, 2, int(self.size / 2))

        target_slic_3 = img_segments_only(
            target_luminance_classic, 4, int(self.size / 4))

        ref_slic_3 = img_segments_only(ref_luminance, 4, int(self.size / 4))

        target_slic_4 = img_segments_only(
            target_luminance_classic, 8, int(self.size / 8))

        ref_slic_4 = img_segments_only(ref_luminance, 8, int(self.size / 8))



        # Applying transformation (To tensor)
        # and replicating tensor for gray scale images
        if self.target_transform:
            target_slic_all = []
            ref_slic_all = []

            target = self.target_transform(target)
            ref_real = self.target_transform(ref_real)

            ref = self.target_transform(ref)

            # Stacking and transform to torch each superpixel maps.

            target_slic_torch = self.target_transform(
                target_slic[:, :, np.newaxis])
            target_slic_torch_2 = self.target_transform(
                target_slic_2[:, :, np.newaxis])
            target_slic_torch_3 = self.target_transform(
                target_slic_3[:, :, np.newaxis])
            target_slic_torch_4 = self.target_transform(
                target_slic_4[:, :, np.newaxis])

            ref_slic_torch = self.target_transform(ref_slic[:, :, np.newaxis])
            ref_slic_torch_2 = self.target_transform(
                ref_slic_2[:, :, np.newaxis])
            ref_slic_torch_3 = self.target_transform(
                ref_slic_3[:, :, np.newaxis])
            ref_slic_torch_4 = self.target_transform(
                ref_slic_4[:, :, np.newaxis])

            target_slic_all.append(target_slic_torch)
            target_slic_all.append(target_slic_torch_2)
            target_slic_all.append(target_slic_torch_3)
            target_slic_all.append(target_slic_torch_4)

            ref_slic_all.append(ref_slic_torch)
            ref_slic_all.append(ref_slic_torch_2)
            ref_slic_all.append(ref_slic_torch_3)
            ref_slic_all.append(ref_slic_torch_4)

            target_luminance_map = self.target_transform(
                target_luminance_map[:, :, np.newaxis])
            target_luminance = self.target_transform(
                target_luminance_classic[:, :, np.newaxis])
            target_chroma = self.target_transform(target_chroma)
            target_luminance_classic_real = self.target_transform(
                target_luminance_classic_real[:, :, np.newaxis])
            target_luminance_classic_real_rep = torch.cat((
                target_luminance_classic_real.float(),
                target_luminance_classic_real.float(),
                target_luminance_classic_real.float()),
                dim=0)
            luminance_replicate_map = torch.cat((
                target_luminance_map.float(),
                target_luminance_map.float(),
                target_luminance_map.float()),
                dim=0)
            luminance_replicate = torch.cat((
                                            target_luminance.float(),
                                            target_luminance.float(),
                                            target_luminance.float()),
                                            dim=0)

            ref_luminance = self.target_transform(
                ref_luminance_classic[:, :, np.newaxis])
            ref_chroma = self.target_transform(ref_chroma)
            ref_luminance_replicate = torch.cat((ref_luminance.float(),
                                                ref_luminance.float(),
                                                ref_luminance.float()),
                                                dim=0)
        """
        Output:
        target: target image rgb,
        luminance_replicate: target grayscale image replicate,
        ref: reference image rdb,
        ref_luminance_replicate: reference grayscale image replicate
        labels_torch: label map target image,
        labels_ref_torch: label map reference image,
        target_luminance: target grayscale image,
        ref_luminance: reference grayscale image,
        ref_real: reference original size. 
        """

        return (target, luminance_replicate, target_chroma, ref,
                ref_luminance_replicate, target_slic_all, ref_slic_all, ref_chroma,
                luminance_replicate_map, target_luminance_classic_real_rep, ref_real)

    def __len__(self):
        return len(self.target_path)


class MyDataTest(Dataset):
    def __init__(self, target_path, ref_sim, slic_target, transform=None,
                 target_transfom=None, slic=True, size=224, color_space='lab'):
        self.target_path = target_path
        self.slic = slic
        self.transform = transform
        self.target_transform = target_transfom
        self.size = size
        self.color_space = color_space
        self.slic_target = slic_target
        self.ref_sim = ref_sim

    def __getitem__(self, index):
        ######## accv_img #############

        target_real = rgb2gray(io.imread('./data/samples/accv/target/'
                                        + str(index) + '_target.png', pilmode='RGB'))
        # Reading target images in RGB
        target = rgb2gray(resize(io.imread('./data/samples/accv/target/'
                                        + str(index) + '_target.png', pilmode='RGB'), (224, 224)))
        ref_real = io.imread('./data/samples/accv/ref/'+str(index)
                            + '_ref.png', pilmode='RGB')
        # Reading ref images in RGB
        ref = resize(io.imread('./data/samples/accv/ref/'
                                + str(index) + '_ref.png', pilmode='RGB'), (224, 224))
        index = index + 1
        if self.color_space == 'lab':
            if np.ndim(target) == 3:

                target_luminance_classic_real = (target_real[:, :, 0])
                target_luminance_classic = (target[:, :, 0])

            else:
                target_luminance_classic_real = (target_real)
                target_luminance_classic = (target)
                target = target[:, :, np.newaxis]
                target_real = target_real[:, :, np.newaxis]

            ref_new_color = color.rgb2lab(ref)
            ref_luminance_classic = (ref_new_color[:, :, 0] / 100.0)
            ref_chroma = ref_new_color[:, :, 1:] / 127.0

            # Luminance remapping
            target_luminance_map = ((
                                    np.std(ref_luminance_classic) / np.std(target_luminance_classic))
                                    * (target_luminance_classic - np.mean(target_luminance_classic))
                                    + np.mean(ref_luminance_classic
                                              ))
            ref_luminance = ref_luminance_classic

        # Calculating superpixel label map for target and reference images (Grayscale)


        target_slic = img_segments_only(target_luminance_classic, 1, self.size)
        ref_slic = img_segments_only(ref_luminance, 1, self.size)

        target_slic_2 = img_segments_only(
            target_luminance_classic, 2, int(self.size / 2))
        ref_slic_2 = img_segments_only(ref_luminance, 2, int(self.size / 2))

        target_slic_3 = img_segments_only(
            target_luminance_classic, 4, int(self.size / 4))
        ref_slic_3 = img_segments_only(ref_luminance, 4, int(self.size / 4))

        target_slic_4 = img_segments_only(
            target_luminance_classic, 8, int(self.size / 8))
        ref_slic_4 = img_segments_only(ref_luminance, 8, int(self.size / 8))

        # Sauvegarde des superpixels
        target_save_2 = img_as_float(resize(
                target_luminance_classic, (len(target_luminance_classic[0]) / 2, len(target_luminance_classic[1]) / 2)))
        target_save_3 = img_as_float(resize(
            target_luminance_classic, (len(target_luminance_classic[0]) / 4, len(target_luminance_classic[1]) / 4)))
        target_save_4 = img_as_float(resize(
            target_luminance_classic, (len(target_luminance_classic[0]) / 8, len(target_luminance_classic[1]) / 8)))
        plt.imsave('./results/Superpixels/SP/targets/' + str(index)
                + '_target.png', mark_boundaries(target_luminance_classic, target_slic))
        plt.imsave('./results/Superpixels/SP/targets/' + str(index)
                + '_target_2.png', mark_boundaries(target_save_2, target_slic_2))
        plt.imsave('./results/Superpixels/SP/targets/' + str(index)
                + '_target_3.png', mark_boundaries(target_save_3, target_slic_3))
        plt.imsave('./results/Superpixels/SP/targets/' + str(index)
                + '_target_4.png', mark_boundaries(target_save_4, target_slic_4))
        
        
        ref_save_2 = img_as_float(resize(
            ref, (len(ref[0]) / 2, len(ref[1]) / 2)))
        ref_save_3 = img_as_float(resize(
            ref, (len(ref[0]) / 4, len(ref[1]) / 4)))
        ref_save_4 = img_as_float(resize(
            ref, (len(ref[0]) / 8, len(ref[1]) / 8)))
        plt.imsave('./results/Superpixels/SP/refs/'+ str(index)
                    + '_ref.png', mark_boundaries(ref, ref_slic))
        plt.imsave('./results/Superpixels/SP/refs/' + str(index)
                    + '_ref_2.png', mark_boundaries(ref_save_2, ref_slic_2))
        plt.imsave('./results/Superpixels/SP/refs/'+ str(index)
                    + '_ref_3.png', mark_boundaries(ref_save_3, ref_slic_3))
        plt.imsave('./results/Superpixels/SP/refs/' + str(index)
                    + '_ref_4.png', mark_boundaries(ref_save_4, ref_slic_4))
            

        # Applying transformation (To tensor)
        # and replicating tensor for gray scale images
        if self.target_transform:
            target_slic_all = []
            ref_slic_all = []
            target = self.target_transform(target)
            target_real = self.target_transform(target_real)
            ref_real = self.target_transform(ref_real)

            ref = self.target_transform(ref)

            target_slic_torch = self.target_transform(
                target_slic[:, :, np.newaxis])
            target_slic_torch_2 = self.target_transform(
                target_slic_2[:, :, np.newaxis])
            target_slic_torch_3 = self.target_transform(
                target_slic_3[:, :, np.newaxis])
            target_slic_torch_4 = self.target_transform(
                target_slic_4[:, :, np.newaxis])

            ref_slic_torch = self.target_transform(ref_slic[:, :, np.newaxis])
            ref_slic_torch_2 = self.target_transform(
                ref_slic_2[:, :, np.newaxis])
            ref_slic_torch_3 = self.target_transform(
                ref_slic_3[:, :, np.newaxis])
            ref_slic_torch_4 = self.target_transform(
                ref_slic_4[:, :, np.newaxis])

            target_slic_all.append(target_slic_torch)
            target_slic_all.append(target_slic_torch_2)
            target_slic_all.append(target_slic_torch_3)
            target_slic_all.append(target_slic_torch_4)

            ref_slic_all.append(ref_slic_torch)
            ref_slic_all.append(ref_slic_torch_2)
            ref_slic_all.append(ref_slic_torch_3)
            ref_slic_all.append(ref_slic_torch_4)

            target_luminance_map = self.target_transform(
                target_luminance_map[:, :, np.newaxis])

            target_luminance = self.target_transform(
                target_luminance_classic[:, :, np.newaxis])

            target_luminance_classic_real = self.target_transform(
                target_luminance_classic_real[:, :, np.newaxis])

            target_luminance_classic_real_rep = torch.cat((
                target_luminance_classic_real.float(),
                target_luminance_classic_real.float(),
                target_luminance_classic_real.float()),
                dim=0)
            luminance_replicate_map = torch.cat((target_luminance_map.float(),
                                                target_luminance_map.float(),
                                                target_luminance_map.float()),
                                                dim=0)
            luminance_replicate = torch.cat((target_luminance.float(),
                                            target_luminance.float(),
                                            target_luminance.float()),
                                            dim=0)

            ref_luminance = self.target_transform(
                ref_luminance_classic[:, :, np.newaxis])
            ref_chroma = self.target_transform(ref_chroma)
            ref_luminance_replicate = torch.cat((ref_luminance.float(),
                                                ref_luminance.float(),
                                                ref_luminance.float()),
                                                dim=0)

        # Output: target: target image rgb,
        # luminance_replicate: target grayscale image replicate,
        # ref: reference image rdb,
        # ref_luminance_replicate: reference grayscale image replicate
        # labels_torch: label map target image,
        # labels_ref_torch: label map reference image,
        # target_luminance: target grayscale image,
        # ref_luminance: reference grayscale image.

        return (target, luminance_replicate, ref, ref_luminance_replicate,
                target_slic_all, ref_slic_all, ref_chroma, luminance_replicate_map,
                target_luminance_classic_real_rep, ref_real)

    def __len__(self):
        _, _, files = next(os.walk("./data/samples/accv/target"))
        file_count = len(files)
        return file_count


class MyDataTest_SH(Dataset):
    def __init__(self, target_path, ref_sim, slic_target, transform=None,
                 target_transfom=None, slic=True, size=224, color_space='lab'):
        self.target_path = target_path
        self.slic = slic
        self.transform = transform
        self.target_transform = target_transfom
        self.size = size
        self.color_space = color_space
        self.slic_target = slic_target
        self.ref_sim = ref_sim

    def __getitem__(self, index):
        ######## accv_img #############

        target_real = rgb2gray(io.imread('./data/samples/accv/target/'
                                        + str(index) + '_target.png', pilmode='RGB'))
        # Reading target images in RGB
        target = rgb2gray(resize(io.imread('./data/samples/accv/target/'
                                        + str(index) + '_target.png', pilmode='RGB'), (224, 224)))
        ref_real = io.imread('./data/samples/accv/ref/'+str(index)
                            + '_ref.png', pilmode='RGB')
        # Reading ref images in RGB
        ref = resize(io.imread('./data/samples/accv/ref/'
                                + str(index) + '_ref.png', pilmode='RGB'), (224, 224))
        index = index + 1
        if self.color_space == 'lab':
            if np.ndim(target) == 3:

                target_luminance_classic_real = (target_real[:, :, 0])
                target_luminance_classic = (target[:, :, 0])

            else:
                target_luminance_classic_real = (target_real)
                target_luminance_classic = (target)
                target = target[:, :, np.newaxis]
                target_real = target_real[:, :, np.newaxis]

            ref_new_color = color.rgb2lab(ref)
            ref_luminance_classic = (ref_new_color[:, :, 0] / 100.0)
            ref_chroma = ref_new_color[:, :, 1:] / 127.0

            # Luminance remapping
            target_luminance_map = ((
                                    np.std(ref_luminance_classic) / np.std(target_luminance_classic))
                                    * (target_luminance_classic - np.mean(target_luminance_classic))
                                    + np.mean(ref_luminance_classic
                                              ))
            ref_luminance = ref_luminance_classic

        # Calculating superpixel label map for target and reference images (Grayscale)

        #Load superpixels

        mat_target_slic = loadmat('./data/samples/accv/labels/target/'+str(index-1)+'_target_1.mat')
        mat_target_slic_2 = loadmat('./data/samples/accv/labels/target/'+str(index-1)+'_target_2.mat')
        mat_target_slic_3 = loadmat('./data/samples/accv/labels/target/'+str(index-1)+'_target_4.mat')
        mat_target_slic_4 = loadmat('./data/samples/accv/labels/target/'+str(index-1)+'_target_8.mat')

        mat_ref_slic = loadmat('./data/samples/accv/labels/ref/'+str(index-1)+'_ref_1.mat')
        mat_ref_slic_2 = loadmat('./data/samples/accv/labels/ref/'+str(index-1)+'_ref_2.mat')
        mat_ref_slic_3 = loadmat('./data/samples/accv/labels/ref/'+str(index-1)+'_ref_4.mat')
        mat_ref_slic_4 = loadmat('./data/samples/accv/labels/ref/'+str(index-1)+'_ref_8.mat')


        target_slic = mat_target_slic['S_SH']
        target_slic = np.int64(target_slic)
        ref_slic = mat_ref_slic['S_SH']
        ref_slic = np.int64(ref_slic)

        target_slic_2 = mat_target_slic_2['S_SH']
        target_slic_2 = np.int64(target_slic_2)
        ref_slic_2 = mat_ref_slic_2['S_SH']
        ref_slic_2 = np.int64(ref_slic_2)

        target_slic_3 = mat_target_slic_3['S_SH']
        target_slic_3 = np.int64(target_slic_3)
        ref_slic_3 = mat_ref_slic_3['S_SH']
        ref_slic_3 = np.int64(ref_slic_3)

        target_slic_4 = mat_target_slic_4['S_SH']
        target_slic_4 = np.int64(target_slic_4)
        ref_slic_4 = mat_ref_slic_4['S_SH']
        ref_slic_4 = np.int64(ref_slic_4)

        # Sauvegarde des superpixels
        target_save_2 = img_as_float(resize(
                target_luminance_classic, (len(target_luminance_classic[0]) / 2, len(target_luminance_classic[1]) / 2)))
        target_save_3 = img_as_float(resize(
            target_luminance_classic, (len(target_luminance_classic[0]) / 4, len(target_luminance_classic[1]) / 4)))
        target_save_4 = img_as_float(resize(
            target_luminance_classic, (len(target_luminance_classic[0]) / 8, len(target_luminance_classic[1]) / 8)))
        plt.imsave('./results/Superpixels/SP/targets/' + str(index)
                + '_target.png', mark_boundaries(target_luminance_classic, target_slic))
        plt.imsave('./results/Superpixels/SP/targets/' + str(index)
                + '_target_2.png', mark_boundaries(target_save_2, target_slic_2))
        plt.imsave('./results/Superpixels/SP/targets/' + str(index)
                + '_target_3.png', mark_boundaries(target_save_3, target_slic_3))
        plt.imsave('./results/Superpixels/SP/targets/' + str(index)
                + '_target_4.png', mark_boundaries(target_save_4, target_slic_4))
        
        
        ref_save_2 = img_as_float(resize(
            ref, (len(ref[0]) / 2, len(ref[1]) / 2)))
        ref_save_3 = img_as_float(resize(
            ref, (len(ref[0]) / 4, len(ref[1]) / 4)))
        ref_save_4 = img_as_float(resize(
            ref, (len(ref[0]) / 8, len(ref[1]) / 8)))
        plt.imsave('./results/Superpixels/SP/refs/'+ str(index)
                    + '_ref.png', mark_boundaries(ref, ref_slic))
        plt.imsave('./results/Superpixels/SP/refs/' + str(index)
                    + '_ref_2.png', mark_boundaries(ref_save_2, ref_slic_2))
        plt.imsave('./results/Superpixels/SP/refs/'+ str(index)
                    + '_ref_3.png', mark_boundaries(ref_save_3, ref_slic_3))
        plt.imsave('./results/Superpixels/SP/refs/' + str(index)
                    + '_ref_4.png', mark_boundaries(ref_save_4, ref_slic_4))
            

        # Applying transformation (To tensor)
        # and replicating tensor for gray scale images
        if self.target_transform:
            target_slic_all = []
            ref_slic_all = []
            target = self.target_transform(target)
            target_real = self.target_transform(target_real)
            ref_real = self.target_transform(ref_real)

            ref = self.target_transform(ref)

            target_slic_torch = self.target_transform(
                target_slic[:, :, np.newaxis])
            target_slic_torch_2 = self.target_transform(
                target_slic_2[:, :, np.newaxis])
            target_slic_torch_3 = self.target_transform(
                target_slic_3[:, :, np.newaxis])
            target_slic_torch_4 = self.target_transform(
                target_slic_4[:, :, np.newaxis])

            ref_slic_torch = self.target_transform(ref_slic[:, :, np.newaxis])
            ref_slic_torch_2 = self.target_transform(
                ref_slic_2[:, :, np.newaxis])
            ref_slic_torch_3 = self.target_transform(
                ref_slic_3[:, :, np.newaxis])
            ref_slic_torch_4 = self.target_transform(
                ref_slic_4[:, :, np.newaxis])

            target_slic_all.append(target_slic_torch)
            target_slic_all.append(target_slic_torch_2)
            target_slic_all.append(target_slic_torch_3)
            target_slic_all.append(target_slic_torch_4)

            ref_slic_all.append(ref_slic_torch)
            ref_slic_all.append(ref_slic_torch_2)
            ref_slic_all.append(ref_slic_torch_3)
            ref_slic_all.append(ref_slic_torch_4)

            target_luminance_map = self.target_transform(
                target_luminance_map[:, :, np.newaxis])

            target_luminance = self.target_transform(
                target_luminance_classic[:, :, np.newaxis])

            target_luminance_classic_real = self.target_transform(
                target_luminance_classic_real[:, :, np.newaxis])

            target_luminance_classic_real_rep = torch.cat((
                target_luminance_classic_real.float(),
                target_luminance_classic_real.float(),
                target_luminance_classic_real.float()),
                dim=0)
            luminance_replicate_map = torch.cat((target_luminance_map.float(),
                                                target_luminance_map.float(),
                                                target_luminance_map.float()),
                                                dim=0)
            luminance_replicate = torch.cat((target_luminance.float(),
                                            target_luminance.float(),
                                            target_luminance.float()),
                                            dim=0)

            ref_luminance = self.target_transform(
                ref_luminance_classic[:, :, np.newaxis])
            ref_chroma = self.target_transform(ref_chroma)
            ref_luminance_replicate = torch.cat((ref_luminance.float(),
                                                ref_luminance.float(),
                                                ref_luminance.float()),
                                                dim=0)

        # Output: target: target image rgb,
        # luminance_replicate: target grayscale image replicate,
        # ref: reference image rdb,
        # ref_luminance_replicate: reference grayscale image replicate
        # labels_torch: label map target image,
        # labels_ref_torch: label map reference image,
        # target_luminance: target grayscale image,
        # ref_luminance: reference grayscale image.

        return (target, luminance_replicate, ref, ref_luminance_replicate,
                target_slic_all, ref_slic_all, ref_chroma, luminance_replicate_map,
                target_luminance_classic_real_rep, ref_real)

    def __len__(self):
        _, _, files = next(os.walk("./data/samples/accv/target"))
        file_count = len(files)
        return file_count
