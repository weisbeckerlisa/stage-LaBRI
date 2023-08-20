
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import numpy as np
from skimage.transform import resize
import torch
import torch.nn as nn
import torchvision.models as models
from utils import get_corresponding_superpixel, get_high_attention_values, img_segments_only, save_attention_maps
from skimage import color, io, transform
from skimage.color import rgb2gray
from skimage.util import img_as_float
from torchvision.models.vgg import VGG
import cv2
import matplotlib.pyplot as plt
from skimage import segmentation
from skimage.morphology import dilation, square

from torchvision import transforms
from PIL import Image

from models import SuperattentionConv, Superfeatures

# Model
class CustomVGG(nn.Module):
    def __init__(self):
        super(CustomVGG, self).__init__()
        vgg = models.vgg19(pretrained=True)
        # Stop avant 1er maxpooling
        self.features = nn.Sequential(*list(vgg.features.children())[:4])

    def forward(self, x):
        x = self.features(x)
        return x

model = CustomVGG()


image_ref_rgb = io.imread('./data/samples/accv/ref/21_ref.png')
image_ref = rgb2gray(image_ref_rgb)
image_target = io.imread('./data/samples/accv/target/21_target.png')

# Get superpixels
num_max_seg = 100
div_resize = 1

# resize images
image_ref_rgb = resize(image_ref_rgb, (224, 224))
image_ref = resize(image_ref, (224, 224))
image_target = resize(image_target, (224, 224))

label_mask_target = img_segments_only(image_target, div_resize, num_max_seg)
label_mask_ref = img_segments_only(image_ref, div_resize, num_max_seg)

print(label_mask_ref.shape)
print(label_mask_ref.dtype)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


image_target_transfo = image_target[..., np.newaxis]
image_target_transfo = np.concatenate([image_target_transfo, image_target_transfo, image_target_transfo], axis=-1)

image_ref_transfo = image_ref[..., np.newaxis]
image_ref_transfo = np.concatenate([image_ref_transfo, image_ref_transfo, image_ref_transfo], axis=-1)

image_ref_transfo = (image_ref_transfo * 255).astype(np.uint8)
image_target_transfo = (image_target_transfo * 255).astype(np.uint8)
image_ref_rgb_transfo = (image_ref_rgb * 255).astype(np.uint8)

image_ref_transfo = Image.fromarray(image_ref_transfo)
image_target_transfo = Image.fromarray(image_target_transfo)
image_ref_rgb_transfo = Image.fromarray(image_ref_rgb_transfo)

image_ref_transfo = transform(image_ref_transfo).unsqueeze(0) 
image_target_transfo = transform(image_target_transfo).unsqueeze(0)
image_ref_rgb_transfo = transform(image_ref_rgb_transfo).unsqueeze(0)

feature_map_ref = model(image_ref_transfo)
feature_map_target = model(image_target_transfo)
feature_map_ref_rgb = model(image_ref_rgb_transfo)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

label_mask_target = torch.from_numpy(label_mask_target).unsqueeze(0)
label_mask_ref = torch.from_numpy(label_mask_ref).unsqueeze(0) 
label_mask_target = label_mask_target.repeat(1, 64, 1, 1)
label_mask_ref = label_mask_ref.repeat(1, 64, 1, 1)

super_attention_model = SuperattentionConv(dims=2, in_channels=64, convolution=False) 
super_attention_model.to(device)

superfeatures = Superfeatures()

# Obtention des super features
cat_encoded_target = superfeatures(feature_map_target, label_mask_target, device)
cat_encoded_ref = superfeatures(feature_map_ref, label_mask_ref, device)
cat_encoded_rgb_ref = superfeatures(feature_map_ref_rgb, label_mask_ref, device)

max_segments = 110
max_segments_ref = 110

wg_ref_batch_transp, attention_batch, batch_alpha = super_attention_model(cat_encoded_target, cat_encoded_ref, label_mask_target,   
                                                                          label_mask_ref, None, max_segments, cat_encoded_rgb_ref,                         
                                                                          max_segments_ref, device)

save_attention_maps(attention_batch, './data/samples/accv/attention_maps/')


attention_map = attention_batch.cpu().detach().numpy()[0]

label_mask_target_cpu = label_mask_target[0,0,:,:].cpu().numpy().astype(int)
label_mask_ref_cpu = label_mask_ref[0,0,:,:].cpu().numpy().astype(int)

red = np.array([1, 0, 0])

# Initialize figures
fig, ax = plt.subplots(1, 3, figsize=(15, 5))  
attention_plot = ax[0].imshow(attention_map, cmap='RdPu')
ax[0].set_title('Attention map')

target_plot = ax[1].imshow(image_target, cmap='gray')
ax[1].set_title('Target image')

ref_plot = ax[2].imshow(image_ref_rgb)
ax[2].set_title('Reference image')

def onclick(event):
    x = int(event.xdata)
    y = int(event.ydata)

    ax[0].clear()
    ax[0].imshow(attention_map, cmap='RdPu')
    ax[0].set_title('Attention map')
                  
    if event.inaxes == ax[1]: #Target image
        label = label_mask_target_cpu[y, x]
        
        index, corresponding_attention_values = get_high_attention_values(attention_map, label)
        index=index[0]
             
        target_img = np.copy(image_target)
        ref_img = np.copy(image_ref_rgb)

        boundary_target = segmentation.mark_boundaries(target_img, label_mask_target_cpu, color=red)    
        
        boundary_target[label_mask_target_cpu == label] = red 

        # Mark interesting superpixels in reference
        mask = np.isin(label_mask_ref_cpu, index)
        boundaries = segmentation.find_boundaries(mask)
        boundary_ref = np.zeros(ref_img.shape[:2])

        for i in range (len(index)):
            intensity = corresponding_attention_values[i]
            boundary_ref[boundaries & (label_mask_ref_cpu == index[i])] = intensity

        cmap = plt.cm.get_cmap('RdPu')

        dilated_boundary = dilation(boundary_ref, square(2))
        colored_boundary_dilated = cmap(dilated_boundary)[:,:,:3]

        mask_contours = np.expand_dims(dilated_boundary > 0, axis=-1)
        mask_3d = np.broadcast_to(mask_contours, colored_boundary_dilated.shape)

        ref_img[mask_3d] = colored_boundary_dilated[mask_3d]

        ax[0].axhline(y=label, color='blue',linewidth=0.5)
        for i in range (len(index)):
            ax[0].axvline(x=index[i], color='green', linewidth=corresponding_attention_values[i])

        # update the image displays
        target_plot.set_data(boundary_target)
        ref_plot.set_data(ref_img)
        fig.canvas.draw()

    elif event.inaxes == ax[0]:

        target_img = np.copy(image_target)
        ref_img = np.copy(image_ref_rgb)

        target_img = segmentation.mark_boundaries(target_img, label_mask_target_cpu, color=red)
        target_img[label_mask_target_cpu == y] = red

        ref_img = segmentation.mark_boundaries(ref_img, label_mask_ref_cpu, color=red)
        ref_img[label_mask_ref_cpu == x] = red

        target_plot.set_data(target_img)
        ref_plot.set_data(ref_img)
        fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()


