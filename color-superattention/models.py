import torch
import torch.nn as nn
from torchvision.models import vgg19_bn
from torch_scatter import scatter_max
from skimage.transform import resize
import kornia


class percep_vgg19_bn(nn.Module):
    """
    Perceptual model.
    Encoder for extracting reference color features. (pre-trained VGG-19).
    """
    def __init__(self):
        super(percep_vgg19_bn, self).__init__()
        self.features = nn.Sequential(
            *list(vgg19_bn(pretrained=True).features.children())[:-1]
        )

    def forward(self, x, device):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {5, 12, 25, 38, 51}:
                results.append(x)
        return results[0], results[1], results[2], results[3], results[4]


#############################################################################
#   Functions and Model Block                                             #
#############################################################################

"""
Encoded superpixel features
"""

class Superfeatures(nn.Module):

    def __init__(self):

        super(Superfeatures, self).__init__()

    def forward(self, input_features_in, label_mask, device):
        """
        :param input_features: Feature tensor [B, C, H, W]
        :param label_mask: int label mask of size [B, C, H, W]
        :return cat_encoded: Staked array's of masked tensor of size [B, C, N_max = M_max]
        """

        batch_size, n_channels = input_features_in.shape[:2]
        sp_maps_expand = label_mask.expand(-1, n_channels, -1, -1).to(device)
        super_features, _ = scatter_max(input_features_in[:, :, 1:-1, 1:-1].flatten(-2).to(device),
                                        sp_maps_expand[:, :, 1:-1, 1:-1].flatten(-2).to(device), dim=2)
        return super_features


id_x = 0

"""
Super attention block
"""


class SuperattentionConv(nn.Module):
    def __init__(self, dims, in_channels, convolution):
        super(SuperattentionConv, self).__init__()
        self.attent_soft = nn.Softmax(dim=dims)
        self.query_conv = nn.Conv1d(in_channels=in_channels,
                                    out_channels=in_channels, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_channels,
                                  out_channels=in_channels, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_channels,
                                    out_channels=in_channels, kernel_size=1)

        #Initialize the weights and biases to identity and zeros
        if not convolution:
            with torch.no_grad():
                self.query_conv.weight.fill_(0)
                self.query_conv.bias.fill_(0)
                torch.diagonal(self.query_conv.weight).fill_(1)

                self.key_conv.weight.fill_(0)
                self.key_conv.bias.fill_(0)
                torch.diagonal(self.key_conv.weight).fill_(1)

                self.value_conv.weight.fill_(0)
                self.value_conv.bias.fill_(0)
                torch.diagonal(self.value_conv.weight).fill_(1)

    def forward(self, cat_encoded_target, cat_encoded_ref, label_mask_target, label_mask_ref, input_features_in, max_segments, cat_encoded_rgb_ref, max_segments_ref, device):
        """
        :param cat_encoded_target: Target segmented features of  size [B ,C, N] where N is the number of target's superpixels.
        :param cat_encoded_ref: Reference segmented features of  size [B, C, M] where M is the number of references superpixels.
        :param label_mask_target: Superpixel's target label map [B, C, H, W].
        :param label_mask_ref: Superpixel's reference label map [B, C, H, W].
        :param input_features_in: Features from previous convolutional block [B, C, H, W] and, it's used for tracking shape for the unpooling features.
        :param max_segments: Maximum number of superpixels.
        :return: wg_ref_feat: Weighted reference features of size [B, N, C] Where N is the number of target's segments.
        """

        batch_alpha = torch.zeros(
            len(cat_encoded_target), max_segments + 1, 1).to(device)
        attention_batch = torch.zeros(
            len(cat_encoded_target), max_segments + 1, max_segments_ref + 1).to(device)
        wg_ref_batch = torch.zeros(
            len(cat_encoded_target),  max_segments + 1, len(cat_encoded_target[0])).to(device)

        for batch_attention in range(len(cat_encoded_target)):  # loop over batches
            eps = 1e-5

            # Normalizing input segmented features
            # Retrieving current superpixels from the target N
            label_target = int(
                torch.max(label_mask_target[batch_attention, :, :])) + 1
            # Retrieving current superpixels from the reference M
            label_ref = int(
                torch.max(label_mask_ref[batch_attention, :, :])) + 1

            encd_target = cat_encoded_target[batch_attention, :, 0:label_target].unsqueeze(
                0).to(device)  # [B, C, N_max = M_max] ----> [B, C, N]
            encd_ref = cat_encoded_ref[batch_attention, :, 0:label_ref].unsqueeze(
                0).to(device)  # [B, C, N_max = M_max] ----> [B, C, M]
            encd_rgb_ref = cat_encoded_rgb_ref[batch_attention, :, 0:label_ref].unsqueeze(
                0).to(device)
            # Feature's normalization
            encd_ref_mean = encd_ref - torch.mean(encd_ref, axis=1)
            encd_target_mean = encd_target - torch.mean(encd_target, axis=1)
            encd_target_norm = encd_target_mean / (torch.norm(encd_target_mean, p=2, keepdim=True,
                                                              dim=1) + eps)  # Normalize encoded features from target
            encd_ref_norm = encd_ref_mean / (torch.norm(encd_ref_mean, p=2, keepdim=True,
                                                        dim=1) + eps)  # Normalize encoded features from reference
            encd_target_norm = self.query_conv(encd_target_norm)
            encd_ref_norm = self.key_conv(encd_ref_norm)

            # Traget transpose.. target [1, C , N_segments] ----> [1, N_segments, C]
            transpose_target = torch.transpose(encd_target_norm, 1, 2)

            # Similarity calculation target: [1, N_segments, C] --- Reference: [1, C, M_segments]
            # Sim: [B, N_segments, M_segments]
            sim = torch.bmm(transpose_target, encd_ref_norm)
            sim_soft = self.attent_soft(sim / 0.01)  # Attention map

            encd_rgb_ref = self.value_conv(encd_rgb_ref)
            transpose_ref = torch.transpose(encd_rgb_ref, 1, 2)

            # Weightening reference enconded features [B, N, C]
            wg_ref_feat = torch.bmm(sim_soft,
                                    transpose_ref)  # Result: sim_soft: [1, N_segments, M_segments] and [1, M_segments, C] = [1, N_segments, C]

            h_sim, w_sim = sim_soft.shape[1:]
            batch_alpha[batch_attention, 0:label_target, :] = 0
            attention_batch[batch_attention,
                            0:h_sim, 0:w_sim] = sim_soft[0, :, :]
            wg_ref_batch[batch_attention, 0:h_sim, :] = wg_ref_feat[0, :, :]

        wg_ref_batch_transp = torch.transpose(wg_ref_batch, 1, 2)  # [B, C, N]
        return wg_ref_batch_transp, attention_batch, batch_alpha


class UnpoolFeatures(nn.Module):

    def __init__(self):

        super(UnpoolFeatures, self).__init__()

    def forward(self, cat_encoded_wg, shape_input_features_in, label_mask, device):
        """
        :param input_features: Feature tensor [B, C, H, W]
        :param cat_encoded_wg: Staked array's of masked tensor of size [B, C, N_max = M_max]
        :param label_mask: int label mask of size [B, C, H, W]
        :return cat_encoded: Staked array's of masked tensor of size [B, C, N_max = M_max]
        """

        batch_size, n_channels, H, W = shape_input_features_in
        sp_maps_expand = label_mask.expand(-1, n_channels, -1, -1).to(device)
        unpool_feat = torch.gather(
            cat_encoded_wg, 2, sp_maps_expand.flatten(-2))
        return torch.reshape(unpool_feat, (batch_size, n_channels, H, W))


"""
Generative model
"""


class gen_color_stride_vgg16(nn.Module):
    def __init__(self, dim):
        super(gen_color_stride_vgg16, self).__init__()

        norm_layer = nn.BatchNorm2d
        num_channel = [64, 128, 256, 512, 1024]
        num_channel_conv = [128, 256, 512, 1024]

        self.model_percep = percep_vgg19_bn()
        self.super_feat = Superfeatures()
        self.Unpool = UnpoolFeatures()

        self.attention_1 = SuperattentionConv(
            dims=dim, in_channels=num_channel[0], convolution=True)
        self.attention_2 = SuperattentionConv(
            dims=dim, in_channels=num_channel[1], convolution=True)
        self.attention_3 = SuperattentionConv(
            dims=dim, in_channels=num_channel[2], convolution=True)
        self.attention_4 = SuperattentionConv(
            dims=dim, in_channels=num_channel[3], convolution=True)

        self.gen1 = nn.Sequential(nn.Conv2d(3, num_channel[0], 3, stride=1, padding=1),
                                  norm_layer(num_channel[0]),
                                  nn.ReLU(True),
                                  nn.Conv2d(
                                      num_channel[0], num_channel[0], 3, stride=1, padding=1),
                                  norm_layer(num_channel[0]),
                                  nn.ReLU(True))

        self.gen1_ref = nn.Sequential(nn.Conv2d(3, num_channel[0], 3, stride=1, padding=1),
                                      norm_layer(num_channel[0]),
                                      nn.ReLU(True),
                                      nn.Conv2d(
                                          num_channel[0], num_channel[0], 3, stride=1, padding=1),
                                      norm_layer(num_channel[0]),
                                      nn.ReLU(True))

        self.down1 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1)

        self.gen2 = nn.Sequential(nn.Conv2d(num_channel[0], num_channel[1], 3, stride=1, padding=1),
                                  norm_layer(num_channel[1]),
                                  nn.ReLU(True),
                                  nn.Conv2d(
                                      num_channel[1], num_channel[1], 3, stride=1, padding=1),
                                  norm_layer(num_channel[1]),
                                  nn.ReLU(True))

        self.gen2_ref = nn.Sequential(nn.Conv2d(num_channel[0], num_channel[1], 3, stride=1, padding=1),
                                      norm_layer(num_channel[1]),
                                      nn.ReLU(True),
                                      nn.Conv2d(
                                          num_channel[1], num_channel[1], 3, stride=1, padding=1),
                                      norm_layer(num_channel[1]),
                                      nn.ReLU(True))

        self.down2 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1)

        self.gen3 = nn.Sequential(nn.Conv2d(num_channel[1], num_channel[2], 3, stride=1, padding=1),
                                  norm_layer(num_channel[2]),
                                  nn.ReLU(True),
                                  nn.Conv2d(
                                      num_channel[2], num_channel[2], 3, stride=1, padding=1),
                                  norm_layer(num_channel[2]),
                                  nn.ReLU(True),
                                  nn.Conv2d(
                                      num_channel[2], num_channel[2], 3, stride=1, padding=1),
                                  norm_layer(num_channel[2]),
                                  nn.ReLU(True))

        self.gen3_ref = nn.Sequential(nn.Conv2d(num_channel[1], num_channel[2], 3, stride=1, padding=1),
                                      norm_layer(num_channel[2]),
                                      nn.ReLU(True),
                                      nn.Conv2d(
                                          num_channel[2], num_channel[2], 3, stride=1, padding=1),
                                      norm_layer(num_channel[2]),
                                      nn.ReLU(True),
                                      nn.Conv2d(
                                          num_channel[2], num_channel[2], 3, stride=1, padding=1),
                                      norm_layer(num_channel[2]),
                                      nn.ReLU(True))

        self.down3 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1)

        self.gen4 = nn.Sequential(nn.Conv2d(num_channel[2], num_channel[3], 3, stride=1, padding=1),
                                  norm_layer(num_channel[3]),
                                  nn.ReLU(True),
                                  nn.Conv2d(
                                      num_channel[3], num_channel[3], 3, stride=1, padding=1),
                                  norm_layer(num_channel[3]),
                                  nn.ReLU(True),
                                  nn.Conv2d(
                                      num_channel[3], num_channel[3], 3, stride=1, padding=1),
                                  norm_layer(num_channel[3]),
                                  nn.ReLU(True))

        self.gen4_ref = nn.Sequential(nn.Conv2d(num_channel[2], num_channel[3], 3, stride=1, padding=1),
                                      norm_layer(num_channel[3]),
                                      nn.ReLU(True),
                                      nn.Conv2d(
                                          num_channel[3], num_channel[3], 3, stride=1, padding=1),
                                      norm_layer(num_channel[3]),
                                      nn.ReLU(True),
                                      nn.Conv2d(
                                          num_channel[3], num_channel[3], 3, stride=1, padding=1),
                                      norm_layer(num_channel[3]),
                                      nn.ReLU(True))

        self.gen4_mid = nn.Sequential(nn.Conv2d(num_channel[3], num_channel[3], 3, stride=1, padding=1),
                                      norm_layer(num_channel[3]),
                                      nn.ReLU(True),
                                      nn.Conv2d(
                                          num_channel[3], num_channel[3], 3, stride=1, padding=1),
                                      norm_layer(num_channel[3]),
                                      nn.ReLU(True),
                                      nn.Conv2d(
                                          num_channel[3], num_channel[3], 3, stride=1, padding=1),
                                      norm_layer(num_channel[3]),
                                      nn.ReLU(True))

        self.gen4_up = nn.ConvTranspose2d(
            num_channel[3], num_channel[2], 4, stride=2, padding=1)

        self.gen5_conv = nn.Conv2d(
            num_channel_conv[2], num_channel[2], 1, stride=1, padding=0)

        self.gen5 = nn.Sequential(nn.Conv2d(num_channel[2], num_channel[2], 3, stride=1, padding=1),
                                  norm_layer(num_channel[2]),
                                  nn.ReLU(True),
                                  nn.Conv2d(
                                      num_channel[2], num_channel[2], 3, stride=1, padding=1),
                                  norm_layer(num_channel[2]),
                                  nn.ReLU(True),
                                  nn.Conv2d(
                                      num_channel[2], num_channel[2], 3, stride=1, padding=1),
                                  norm_layer(num_channel[2]),
                                  nn.ReLU(True))

        self.gen5_up = nn.Sequential(nn.ConvTranspose2d(num_channel[2], num_channel[1], 4, stride=2, padding=1),
                                     nn.ReLU(True))

        self.gen6_conv = nn.Conv2d(
            num_channel_conv[1], num_channel[1], 1, stride=1, padding=0)

        self.gen6 = nn.Sequential(nn.Conv2d(num_channel[1], num_channel[1], 3, stride=1, padding=1),
                                  norm_layer(num_channel[1]),
                                  nn.ReLU(True),
                                  nn.Conv2d(
                                      num_channel[1], num_channel[1], 3, stride=1, padding=1),
                                  norm_layer(num_channel[1]),
                                  nn.ReLU(True),
                                  nn.Conv2d(
                                      num_channel[1], num_channel[1], 3, stride=1, padding=1),
                                  norm_layer(num_channel[1]),
                                  nn.ReLU(True))

        self.gen6_up = nn.Sequential(nn.ConvTranspose2d(num_channel[1], num_channel[0], 4, stride=2, padding=1),
                                     nn.ReLU(True))

        self.gen7_conv = nn.Conv2d(
            num_channel_conv[0], num_channel[0], 1, stride=1, padding=0)

        self.gen7 = nn.Sequential(nn.Conv2d(num_channel[0], num_channel[0], 3, stride=1, padding=1),
                                  norm_layer(num_channel[0]),
                                  nn.ReLU(True),
                                  nn.Conv2d(
                                      num_channel[0], num_channel[0], 3, stride=1, padding=1),
                                  norm_layer(num_channel[0]),
                                  nn.ReLU(True),
                                  nn.Conv2d(
                                      num_channel[0], num_channel[0], 3, stride=1, padding=1),
                                  norm_layer(num_channel[0]),
                                  nn.ReLU(True))

        self.out = nn.Sequential(nn.Conv2d(num_channel[0], 2, 1, padding=0, dilation=1, stride=1),
                                 nn.Tanh())

    def forward(self, target, ref, luminance_target, label_mask_x, label_mask_ref, target_map, x_real, img_ref_color_norm, device, mode=None):

        feat1_pred, feat2_pred, feat3_pred, feat4_pred, feat5_pred = self.model_percep(
            img_ref_color_norm, device)

        label_max_1 = torch.max(label_mask_x[0])
        label_max_2 = torch.max(label_mask_x[1])
        label_max_3 = torch.max(label_mask_x[2])
        label_max_4 = torch.max(label_mask_x[3])

        label_mask_ref_1 = torch.max(label_mask_ref[0])
        label_mask_ref_2 = torch.max(label_mask_ref[1])
        label_mask_ref_3 = torch.max(label_mask_ref[2])
        label_mask_ref_4 = torch.max(label_mask_ref[3])

        # Reference RGB image
        block1_rgb_ref = feat1_pred
        block2_rgb_ref = feat2_pred
        block3_rgb_ref = feat3_pred
        block4_rgb_ref = feat4_pred

        # Reference image path
        block1_ref = self.gen1_ref(ref)
        block1_down_ref = self.down1(block1_ref)

        block2_ref = self.gen2_ref(block1_down_ref)
        block2_down_ref = self.down2(block2_ref)

        block3_ref = self.gen3_ref(block2_down_ref)
        block3_down_ref = self.down3(block3_ref)

        block4_ref = self.gen4_ref(block3_down_ref)

        # ENCODER
        # Target image path
        block1 = self.gen1(target_map)
        # Super-Attention
        encd_feat1_t = self.super_feat(block1, label_mask_x[0], device)
        encd_feat1_r = self.super_feat(block1_ref, label_mask_ref[0], device)
        encd_feat1_rgb_r = self.super_feat(
            block1_rgb_ref, label_mask_ref[0], device)
        superattention_1, sim_matrix_1, alpha_1 = self.attention_1(
            encd_feat1_t, encd_feat1_r, label_mask_x[0], label_mask_ref[0], block1, label_max_1, encd_feat1_rgb_r, label_mask_ref_1, device)
        wg_1 = self.Unpool(superattention_1, block1.shape,
                           label_mask_x[0], device)
        block1 = wg_1 + block1
        block1_down = self.down1(block1)
        superattention_12 = (sim_matrix_1).to(device=device, dtype=torch.float)

        block2 = self.gen2(block1_down)
        encd_feat2_t = self.super_feat(block2, label_mask_x[1], device)
        encd_feat2_r = self.super_feat(block2_ref, label_mask_ref[1], device)
        encd_feat2_rgb_r = self.super_feat(
            block2_rgb_ref, label_mask_ref[1], device)
        superattention_2, sim_matrix_2, alpha_2 = self.attention_2(
            encd_feat2_t, encd_feat2_r, label_mask_x[1], label_mask_ref[1], block2, label_max_2, encd_feat2_rgb_r, label_mask_ref_2, device)
        wg_2 = self.Unpool(superattention_2, block2.shape,
                           label_mask_x[1], device)
        block2 = wg_2 + block2
        block2_down = self.down2(block2)
        superattention_22 = (sim_matrix_2).to(device=device, dtype=torch.float)

        block3 = self.gen3(block2_down)
        encd_feat3_t = self.super_feat(block3, label_mask_x[2], device)
        encd_feat3_r = self.super_feat(block3_ref, label_mask_ref[2], device)
        encd_feat3_rgb_r = self.super_feat(
            block3_rgb_ref, label_mask_ref[2], device)
        superattention_3, sim_matrix_3, alpha_3 = self.attention_3(
            encd_feat3_t, encd_feat3_r, label_mask_x[2], label_mask_ref[2], block3, label_max_3, encd_feat3_rgb_r, label_mask_ref_3, device)
        wg_3 = self.Unpool(superattention_3, block3.shape,
                           label_mask_x[2], device)
        block3 = wg_3 + block3
        block3_down = self.down3(block3)
        superattention_32 = (sim_matrix_3).to(device=device, dtype=torch.float)

        block4 = self.gen4(block3_down)
        encd_feat4_t = self.super_feat(block4, label_mask_x[3], device)
        encd_feat4_r = self.super_feat(block4_ref, label_mask_ref[3], device)
        encd_feat4_rgb_r = self.super_feat(
            block4_rgb_ref, label_mask_ref[3], device)
        superattention_4, sim_matrix_4, alpha_4 = self.attention_4(
            encd_feat4_t, encd_feat4_r, label_mask_x[3], label_mask_ref[3], block4, label_max_4, encd_feat4_rgb_r, label_mask_ref_4, device)
        wg_4 = self.Unpool(superattention_4, block4.shape,
                           label_mask_x[3], device)
        block4 = wg_4 + block4
        block4 = self.gen4_mid(block4)
        superattention_42 = (sim_matrix_4).to(device=device, dtype=torch.float)
        
        # Decoder and skip connections
        block4_up = self.gen4_up(block4)  # Block 6

        block5_skip = torch.cat((block4_up, block3),
                                dim=1)  # Concatenating over channels [B,256,H/4, W/4] ---> [B, 2*256, H/4, W/4]
        # [B,2*256,H/4, W/4] ---> [B, 256, H/4, W/4]
        block5_conv = self.gen5_conv(block5_skip)
        block5 = self.gen5(block5_conv)  # [B,256,H/4, W/4]
        block5_up = self.gen5_up(block5)  # Block 7

        block6_skip = torch.cat((block5_up, block2),
                                dim=1)  # Concatenating over channels [B,128,H/4, W/4] ---> [B, 3*128, H/4, W/4]
        # [B,2*128,H/4, W/4] ---> [B, 128, H/4, W/4]
        block6_conv = self.gen6_conv(block6_skip)
        block6 = self.gen6(block6_conv)
        block6_up = self.gen6_up(block6)  # Block 8

        block7_skip = torch.cat((block6_up, block1),
                                dim=1)  # Concatenating over channels [B,64,H/4, W/4] ---> [B, 3*64, H/4, W/4]
        # [B,3*64,H/4, W/4] ---> [B, 64, H/4, W/4]
        block7_conv = self.gen7_conv(block7_skip)
        block7 = self.gen7(block7_conv)
        out_reg = self.out(block7)

        if mode == 'train':
            out_unorm = (out_reg * 127.0)
            pred_lab_torch = torch.cat(
                (luminance_target[:, 0, :, :].unsqueeze(1), out_reg), dim=1)
            luminance_unnorm = (luminance_target[:, 0, :, :]) * 100.0
            pred_unorm_lab_torch = torch.cat(
                (luminance_unnorm.unsqueeze(1), out_unorm), dim=1)
            pred_rgb = kornia.color.lab_to_rgb(pred_unorm_lab_torch, clip=True)

        # Interpolating for having original size
        else:
            c, h, w = x_real[:, 0, :, :].shape
            print(c, h, w)

            out_unorm = (out_reg * 127.0)
            predicted_ab = resize(out_unorm[0, :, :, :].cpu().numpy(), (2, h, w))
            luminance_unnorm = (x_real[:, 0, :, :] * 100.0)
            predicted_ab = torch.tensor(predicted_ab).to(device)
            pred_lab_torch = torch.cat(
                (luminance_target[:, 0, :, :].unsqueeze(1), out_reg), dim=1)
            pred_unorm_lab_torch = torch.cat(
                (luminance_unnorm.unsqueeze(1), predicted_ab.unsqueeze(0)), dim=1)
            pred_rgb = kornia.color.lab_to_rgb(pred_unorm_lab_torch, clip=True)

        return out_reg, pred_lab_torch, pred_rgb, (superattention_12, superattention_22, superattention_32, superattention_42)
