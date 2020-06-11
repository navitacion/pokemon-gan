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


@st.cache
def model_init(weight_path, exp):
    if 'SAGAN' in exp:
        G = models.Generator_sagan(z_dim=Z_DIM, image_size=64)
    else:
        G = models.Generator(z_dim=Z_DIM, image_size=64)
    G.load_state_dict(torch.load(weight_path, map_location=device))
    return G


BATCH_SIZE = 8
Z_DIM = 500

st.title('Pokemon GAN')

exp = st.selectbox('Select Exp', ('GAN_01', 'SAGAN_01'))

# if exp == 'GAN_01':
#     Z_DIM = 80
# elif exp == 'SAGAN_01':
#     Z_DIM = 500

epoch = st.slider('Epoch', min_value=0, max_value=5000, step=50)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


weight_path = f'./weights/{exp}_netG_epoch_{epoch}.pth'
G = model_init(weight_path, exp)

fixed_z = torch.randn(BATCH_SIZE, Z_DIM, 1, 1)
d_out = G(fixed_z)

fig, axes = plt.subplots(ncols=5, nrows=4, figsize=(16, 16))
for i, ax in enumerate(axes.ravel()):
    input_z = torch.randn(1, Z_DIM, 1, 1)
    d_out = G(input_z)
    out = d_out[0]
    out = out.detach().permute(1, 2, 0).numpy()

    # Reverse Normalize
    _max, _min = out.max(), out.min()
    out = (out - _min) * 255 / (_max - _min)
    # out = out * 0.5 + 0.5
    out = out.astype(int)

    ax.imshow(out)
    ax.axis('off')

plt.tight_layout()

st.subheader('Generative Pokemon Images')
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
