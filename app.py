import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
from torch import nn, optim
import streamlit as st

from src import models


BATCH_SIZE = 8
Z_DIM = 80

st.title('Pokemon GAN')

epoch = st.slider('Epoch', min_value=0, max_value=100)
# torch.manual_seed(seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@st.cache
def model_init(weight_path):
    G = models.Generator(z_dim=20, image_size=256)
    G.load_state_dict(torch.load(weight_path, map_location=device))
    return G


exp = 'GAN_01'
weight_path = f'./weights/{exp}_netG_epoch_{epoch}.pth'
G = model_init(weight_path)

fixed_z = torch.randn(BATCH_SIZE, Z_DIM, 1, 1)
d_out = G(fixed_z)

fig, axes = plt.subplots(ncols=4, nrows=2)
for i, ax in enumerate(axes.ravel()):
    input_z = torch.randn(1, Z_DIM, 1, 1)
    d_out = G(input_z)
    out = d_out[0]
    out = out.detach().permute(1, 2, 0).numpy()

    # Reverse Normalize
    _max, _min = out.max(), out.min()
    out = (out - _min) * 255 / (_max - _min)
    out = out.astype(int)

    ax.imshow(out)
    ax.axis('off')

plt.tight_layout()

st.subheader('Generative')
st.pyplot()

# img_transform = fake_images[0].detach().permute(1, 2, 0).numpy()
#
# _max, _min = img_transform.max(), img_transform.min()
# img_transform = (img_transform - _min) * 255 / (_max - _min)
# img_transform = img_transform.astype(int)
#
#
# st.subheader('Generate')
# st.image(img_transform)
