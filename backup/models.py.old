import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import torchvision.utils as vutils

# --- Assumed Local Imports (User's files) ---
import DBAdapters as dba

# ==============================================================================
# 0. Helper Functions & Model Definitions
# ==============================================================================


def conv_cond_concat(x, y):
    """Concatenates a feature map x with a conditional vector y."""
    y_shape = y.shape
    # Reshape and tile y to match the spatial dimensions of x
    y_tiled = y.view(y_shape[0], y_shape[1], 1, 1).repeat(1, 1, x.shape[2], x.shape[3])
    return torch.cat([x, y_tiled], 1)


def conv_prev_concat(x, prev):
    """Concatenates a feature map x with a previous bar's feature map prev."""
    return torch.cat([x, prev], 1)


# --- Generator ---
class Generator(nn.Module):
    def __init__(self, pitch_range):
        super(Generator, self).__init__()

        # generator feature dimension
        self.gf_dim = 64
        # chord vector dimension
        self.y_dim = 13
        # n of filters in the convolution layer
        self.n_channel = 256

        # --- Refactored Batch Norm Layers (Best Practice) ---
        self.bn1d_h0 = nn.BatchNorm1d(1024)
        self.bn1d_h1 = nn.BatchNorm1d(self.gf_dim * 2 * 2 * 1)
        self.bn2d_h0_prev = nn.BatchNorm2d(16)
        self.bn2d_h1_prev = nn.BatchNorm2d(16)
        self.bn2d_h2_prev = nn.BatchNorm2d(16)
        self.bn2d_h3_prev = nn.BatchNorm2d(16)
        self.bn2d_h2 = nn.BatchNorm2d(pitch_range)
        self.bn2d_h3 = nn.BatchNorm2d(pitch_range)
        self.bn2d_h4 = nn.BatchNorm2d(pitch_range)

        # --- Linear Layers ---
        self.linear1 = nn.Linear(100 + self.y_dim, 1024)  # noise_dim + y_dim
        self.linear2 = nn.Linear(1024 + self.y_dim, self.gf_dim * 2 * 2 * 1)

        # --- Conditioner Convolutional Layers (Downsampling) ---
        # turn a matrix into a 13
        self.h0_prev = nn.Conv2d(1, 16, kernel_size=(1, pitch_range), stride=(1, 2))
        self.h1_prev = nn.Conv2d(16, 16, kernel_size=(2, 1), stride=(2, 2))
        self.h2_prev = nn.Conv2d(16, 16, kernel_size=(2, 1), stride=(2, 2))
        self.h3_prev = nn.Conv2d(16, 16, kernel_size=(2, 1), stride=(2, 2))

        # --- Main Transposed Convolutional Layers (Upsampling) ---
        # Inputs channels = reshaped_h1 + y_dim + prev_bar_features
        # in_channel=128+13+16=157
        self.h1_deconv = nn.ConvTranspose2d(
            pitch_range + self.y_dim + 16,
            pitch_range,
            kernel_size=(2, 1),
            stride=(2, 2),
        )
        self.h2_deconv = nn.ConvTranspose2d(
            pitch_range + self.y_dim + 16,
            pitch_range,
            kernel_size=(2, 1),
            stride=(2, 2),
        )
        self.h3_deconv = nn.ConvTranspose2d(
            pitch_range + self.y_dim + 16,
            pitch_range,
            kernel_size=(2, 1),
            stride=(2, 2),
        )
        self.h4_deconv = nn.ConvTranspose2d(
            pitch_range + self.y_dim + 16,
            1,
            kernel_size=(1, pitch_range),
            stride=(1, 2),
        )

    def forward(self, z, prev_x, y, batch_size, pitch_range):
        # --- Conditioner Path (Process previous bar) ---

        # effectively downsample to the point for concatenating and till the shape if 72
        h0_prev = F.leaky_relu(self.bn2d_h0_prev(self.h0_prev(prev_x)), 0.2)
        h1_prev = F.leaky_relu(self.bn2d_h1_prev(self.h1_prev(h0_prev)), 0.2)
        h2_prev = F.leaky_relu(self.bn2d_h2_prev(self.h2_prev(h1_prev)), 0.2)
        h3_prev = F.leaky_relu(self.bn2d_h3_prev(self.h3_prev(h2_prev)), 0.2)
        # [72, 16, 16, 1]
        # [72, 16, 8, 1]
        # [72, 16, 4, 1]
        # [72, 16, 2, 1])

        # --- Main Generator Path ---
        # yb = ...: The 1-D chord condition vector y is reshaped into a 4-D tensor
        # (batch_size, y_dim, 1, 1) so it can be concatenated with the 4-D feature maps produced by the convolutional layers.
        yb = y.view(batch_size, self.y_dim, 1, 1)
        # random noise is concatenated with the chord condition
        z_cond = torch.cat([z, y], 1)

        # START MAIN generator
        # first layer
        # z_cond = (noise + chord) is than passed trough the first nn
        # apply first layer + normalize + apply relu
        h0 = F.relu(self.bn1d_h0(self.linear1(z_cond)))
        # another concatenation with the chord condition
        h0_cond = torch.cat([h0, y], 1)  # (72,1037)
        # second layer
        h1 = F.relu(self.bn1d_h1(self.linear2(h0_cond)))
        # The output is reshaped into a 4-D tensor to be compatible with the transposed convolution layers
        h1_reshaped = h1.view(batch_size, self.gf_dim * 2, 2, 1)
        # The reshaped tensor is concatenated with the 1-D chord condition (yb).
        h1_cond = conv_cond_concat(h1_reshaped, yb)
        # This tensor is then concatenated with the output of the FINAL conditioner layer (h3_prev),
        # which contains information about the previous bar.
        h1_full = conv_prev_concat(h1_cond, h3_prev)

        h2 = F.relu(self.bn2d_h2(self.h1_deconv(h1_full)))
        h2_cond = conv_cond_concat(h2, yb)
        h2_full = conv_prev_concat(h2_cond, h2_prev)

        h3 = F.relu(self.bn2d_h3(self.h2_deconv(h2_full)))
        h3_cond = conv_cond_concat(h3, yb)
        h3_full = conv_prev_concat(h3_cond, h1_prev)

        h4 = F.relu(self.bn2d_h4(self.h3_deconv(h3_full)))
        h4_cond = conv_cond_concat(h4, yb)
        h4_full = conv_prev_concat(h4_cond, h0_prev)

        g_x = torch.sigmoid(self.h4_deconv(h4_full))
        return g_x


# TODO: da qui in poi bisogna rivedere
# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self, pitch_range):
        super(Discriminator, self).__init__()
        self.df_dim = 64
        self.y_dim = 13

        # --- Refactored Batch Norm Layers (Best Practice) ---
        self.bn2d_h1 = nn.BatchNorm2d(self.df_dim + self.y_dim)
        self.bn1d_h2 = nn.BatchNorm1d(1024)

        # --- Convolutional Layers ---
        self.h0_conv = nn.Conv2d(
            1 + self.y_dim, 1 + self.y_dim, kernel_size=(2, pitch_range), stride=(2, 2)
        )
        self.h1_conv = nn.Conv2d(
            1 + self.y_dim + self.y_dim,
            self.df_dim + self.y_dim,
            kernel_size=(4, 1),
            stride=(2, 2),
        )

        # --- Linear Layers ---
        # Input size for linear1 is calculated from the output of h1_conv flattened + y_dim
        # h1_conv output spatial size is approx (3,1), channels are (df_dim + y_dim) -> 77 * 3 * 1 = 231
        self.linear1 = nn.Linear(231 + self.y_dim, 1024)
        self.linear2 = nn.Linear(1024 + self.y_dim, 1)

    def forward(self, x, y, batch_size, pitch_range):
        yb = y.view(batch_size, self.y_dim, 1, 1)
        x_cond = conv_cond_concat(x, yb)

        h0 = F.leaky_relu(self.h0_conv(x_cond), 0.2)
        feature_map = h0  # This is the intermediate feature map for feature matching
        h0_cond = conv_cond_concat(h0, yb)

        h1 = F.leaky_relu(self.bn2d_h1(self.h1_conv(h0_cond)), 0.2)
        h1_flat = h1.view(batch_size, -1)
        h1_cond = torch.cat([h1_flat, y], 1)

        h2 = F.leaky_relu(self.bn1d_h2(self.linear1(h1_cond)), 0.2)
        h2_cond = torch.cat([h2, y], 1)

        logits = self.linear2(h2_cond)
        sigmoid_out = torch.sigmoid(logits)

        return sigmoid_out, logits.squeeze(), feature_map


# --- Conditioner ---
class Conditioner(nn.Module):
    def __init__(self, pitch_range):
        super(Conditioner, self).__init__()
        # --- Batch Norm Layers ---
        self.bn2d_h0 = nn.BatchNorm2d(16)
        self.bn2d_h1 = nn.BatchNorm2d(16)
        self.bn2d_h2 = nn.BatchNorm2d(16)
        self.bn2d_h3 = nn.BatchNorm2d(16)

        # --- Convolutional Layers (Downsampling) ---
        self.h0_conv = nn.Conv2d(1, 16, kernel_size=(1, pitch_range), stride=(1, 2))
        self.h1_conv = nn.Conv2d(16, 16, kernel_size=(2, 1), stride=(2, 2))
        self.h2_conv = nn.Conv2d(16, 16, kernel_size=(2, 1), stride=(2, 2))
        self.h3_conv = nn.Conv2d(16, 16, kernel_size=(2, 1), stride=(2, 2))

    def forward(self, prev_x):
        """
        Processes the previous melody bar to extract conditional features.
        Returns a list of feature maps from each layer.
        """
        h0 = F.leaky_relu(self.bn2d_h0(self.h0_conv(prev_x)), 0.2)
        h1 = F.leaky_relu(self.bn2d_h1(self.h1_conv(h0)), 0.2)
        h2 = F.leaky_relu(self.bn2d_h2(self.h2_conv(h1)), 0.2)
        h3 = F.leaky_relu(self.bn2d_h3(self.h3_conv(h2)), 0.2)
        return [h0, h1, h2, h3]
