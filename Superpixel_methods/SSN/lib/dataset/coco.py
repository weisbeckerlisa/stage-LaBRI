import os
import numpy as np
import torch
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor
from skimage.color import rgb2lab
from PIL import Image
import matplotlib.pyplot as plt

class COCOData:
    def __init__(self, root, split="train", color_transforms=None, geo_transforms=None):
        self.img_dir = os.path.join(root, 'images', split)
        self.seg_dir = os.path.join(root, 'annotations', f'panoptic_{split}')
        
        self.img_ids = [f.split('.')[0] for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        
        self.color_transforms = color_transforms
        self.geo_transforms = geo_transforms
        self.resize_transform = Compose([ToPILImage(), Resize((321, 481), interpolation=Image.NEAREST), ToTensor()])
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, f'{img_id}.jpg')
        seg_path = os.path.join(self.seg_dir, f'{img_id}.png')
        
        img = plt.imread(img_path).astype(np.uint8)
        seg = (plt.imread(seg_path) * 255).astype(np.uint8)

        # Resize and normalize
        img = self.resize_transform(img)
        img = np.array(img) / 255.0  # Convert PIL Image to numpy array and normalize

        # Convert to LAB
        #print("img shape:", img.shape)
        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis = 0)
        img = img.transpose((1, 2, 0))
        #print("img shape transpose:", img.shape)
        img = rgb2lab(img)

        seg = self.resize_transform(seg)
        seg = np.array(seg)
        seg = seg.transpose((1, 2, 0))

        seg = self.convert_segments_to_labels(seg)
        

        #print("img shape:", img.shape)
        #print("seg shape:", seg.shape)

        if self.color_transforms is not None:
            img = self.color_transforms(img)

        if self.geo_transforms is not None:
            img, seg = self.geo_transforms([img, seg])

        #print("seg avant convert:", seg.shape)

        seg = self.convert_to_onehot(seg)
        seg = torch.from_numpy(seg)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)

        #print("seg return:", seg.shape)
        #print("seg reshape:", seg.reshape(50,-1).float().shape)

        return img, seg.reshape(50, -1).float()

    def __len__(self):
        return len(self.img_ids)

    def convert_segments_to_labels(self, seg_img):
        seg_img_encoded = seg_img[..., 0] * 256 * 256 + seg_img[..., 1] * 256 + seg_img[..., 2]
        labels = np.zeros(seg_img_encoded.shape, dtype=np.int32)

        unique_classes = np.unique(seg_img_encoded)
        for i, unique_class in enumerate(unique_classes):
            if i >= 50: break
            labels[seg_img_encoded == unique_class] = i

        return labels

    def convert_to_onehot(self, seg_img):
        labels = np.zeros((1, 50, seg_img.shape[0], seg_img.shape[1]), dtype=np.float32)
        unique_classes = np.unique(seg_img)
        for i, unique_class in enumerate(unique_classes):
            if i >= 50: break
            labels[:, i, :, :] = (seg_img == unique_class).astype(np.float32)
        
        return labels