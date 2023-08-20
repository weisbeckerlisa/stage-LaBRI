import numpy as np
import torch
import training
import testing
from data import MyDataTrain, DataLoader, MyDataTest
from models import gen_color_stride_vgg16
from torchvision.transforms import ToTensor
import torch.optim as optim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch.nn as nn
import warnings
import argparse
""" import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True) """


warnings.filterwarnings('ignore')


# Selecting GPU device
device = torch.device('cpu')

train_data = []
valid_data = []
losses = []

#############################################################################
#  Main Block                                                               #
#############################################################################

mode = 'test'

analogies_val_new = np.array((5000, 5))

# Load similarity matrix on input images (Target ---> Ref)
analogies_train = np.load('./data/analogies/train/analogies_2_8k.npy')
analogies_val = np.load('./data/analogies/val/analogies_val_30k.npy')


if mode == 'train':
    dataset_train = MyDataTrain(target_path=analogies_train, ref_sim=analogies_train, slic_target=None,
                                  transform=None, target_transfom=ToTensor())
    dataset_val = MyDataTrain(target_path=analogies_val[0:5000], ref_sim=analogies_val[0:5000], slic_target=None,
                                transform=None, target_transfom=ToTensor())

    loader_train = DataLoader(
        dataset_train, batch_size=4, num_workers=4, shuffle=True, pin_memory=True)
    loader_val = DataLoader(dataset_val, batch_size=4,
                            num_workers=4, shuffle=False, pin_memory=True)
    # Load model and initializing VGG 19 weights and bias
    model = gen_color_stride_vgg16(dim=2)
    model = model.to(device=device)
    model.train()

    # Loss and optimizer
    learning_rate = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_feat = nn.SmoothL1Loss()
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type='vgg').to(device=device)
    best_loss = 1
    num_epochs = 100

    if __name__ == '__main__':
        training.training_color(loader_train, loader_val, model,
                                num_epochs, device, optimizer, loss_feat, lpips, mode)
elif mode == 'test':
    dataset_train = MyDataTest(target_path=analogies_train, ref_sim=analogies_train, slic_target=None,
                                 transform=None, target_transfom=ToTensor())
    dataset_val = MyDataTest(target_path=analogies_val[0:5000], ref_sim=analogies_val[0:5000], slic_target=None,
                               transform=None, target_transfom=ToTensor())

    loader_train = DataLoader(
        dataset_train, batch_size=8, num_workers=8, shuffle=True, pin_memory=True)
    loader_val = DataLoader(dataset_val, batch_size=1,
                            num_workers=8, shuffle=False, pin_memory=True)

    # Load models.
    PATH_model = './models/super_attent_v1_res_connection/best_model_1114900.pt'

    # Load model and initializing VGG 19 weights and bias
    model = gen_color_stride_vgg16(dim=2)
    model.load_state_dict(torch.load(
        PATH_model, map_location=device)['state_dict'])
    model.to(device=device)
    model.eval()

    for param_color in model.parameters():
        param_color.requires_grad = False

    if __name__ == '__main__':
        save_maps = False
        parser = argparse.ArgumentParser()
        parser.add_argument("--save_attention_maps", help="activates the saving of attention maps", action="store_true")
        args = parser.parse_args()
        if args.save_attention_maps:
            print("attention maps will be saved\n")
            save_maps = True

        print("Launching the colorization...\n")
        testing.testing_color(loader_val, model, device, mode, save_maps)
