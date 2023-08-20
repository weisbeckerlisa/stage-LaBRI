import math
import numpy as np
import torch
import os

from skimage.color import rgb2lab
from skimage.segmentation._slic import _enforce_label_connectivity_cython

from lib.ssn.ssn import sparse_ssn_iter


@torch.no_grad()
def inference(image, nspix, n_iter, fdim=None, color_scale=0.26, pos_scale=2.5, weight=None, enforce_connectivity=True):
    """
    generate superpixels

    Args:
        image: numpy.ndarray
            An array of shape (h, w, c)
        nspix: int
            number of superpixels
        n_iter: int
            number of iterations
        fdim (optional): int
            feature dimension for supervised setting
        color_scale: float
            color channel factor
        pos_scale: float
            pixel coordinate factor
        weight: state_dict
            pretrained weight
        enforce_connectivity: bool
            if True, enforce superpixel connectivity in postprocessing

    Return:
        labels: numpy.ndarray
            An array of shape (h, w)
    """
    if weight is not None:
        from model import SSNModel
        model = SSNModel(fdim, nspix, n_iter).to("cuda")
        model.load_state_dict(torch.load(weight))
        model.eval()
    else:
        print(2)
        model = lambda data: sparse_ssn_iter(data, nspix, n_iter)

    height, width = image.shape[:2]

    nspix_per_axis = int(math.sqrt(nspix))
    pos_scale = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)

    coords = torch.stack(torch.meshgrid(torch.arange(height, device="cuda"), torch.arange(width, device="cuda")), 0)
    coords = coords[None].float()

    image = rgb2lab(image)
    image = torch.from_numpy(image).permute(2, 0, 1)[None].to("cuda").float()

    inputs = torch.cat([color_scale*image, pos_scale*coords], 1)

    _, H, _ = model(inputs)

    labels = H.reshape(height, width).to("cpu").detach().numpy()

    if enforce_connectivity:
        segment_size = height * width / nspix
        min_size = int(0.06 * segment_size)
        max_size = int(3.0 * segment_size)
        labels = _enforce_label_connectivity_cython(
            labels[None], min_size, max_size)[0]

    return labels


if __name__ == "__main__":

    print(43)
    import time
    import argparse
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, type=str, help="/path/to/image")
    parser.add_argument("--folder", default=None, type=str, help="/path/to/folder containing images")
    parser.add_argument("--weight", default=None, type=str, help="/path/to/pretrained_weight")
    parser.add_argument("--fdim", default=20, type=int, help="embedding dimension")
    parser.add_argument("--niter", default=10, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=100, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    args = parser.parse_args()

    np_range = [800, 600, 450, 325, 275, 200, 150, 100, 50, 25]
    #np_range = [800, 600, 450, 300, 175, 100, 50, 25]
    #np_range = [500,400,325,275,200,150]

    if (args.image!=None):
        image = plt.imread(args.image)
        s = time.time()
        print(2)
        label = inference(image, args.nspix, args.niter, args.fdim, args.color_scale, args.pos_scale, args.weight)
        print(f"time {time.time() - s}sec")
        plt.imsave("results_16004_" + str(args.pos_scale) + "_" + str(args.color_scale) + ".png", mark_boundaries(image, label))
    elif (args.folder!=None):
        ext = (".png", ".jpg", ".jpeg")
        for images in os.listdir(args.folder):
            s = time.time()
            image = plt.imread(args.folder+'/'+images)
            print(images)
            for nsp in np_range:
                print(nsp)
                label = inference(image, nsp, args.niter, args.fdim, args.color_scale, args.pos_scale, args.weight)
                print(f"time {time.time() - s}sec")
                plt.imsave("./results_newmodel/"+images[0:-4]+'_'+str(nsp)+".png", mark_boundaries(image, label))
                res_dest = './res_deep/'+images[0:-4]+'_'+str(nsp)+'_label.npy'
                np.save(res_dest, label)
    else:
        print("No image or folder given")
