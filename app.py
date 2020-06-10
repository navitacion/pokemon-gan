import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
from torch import nn, optim
import streamlit as st

from src import utils, models


st.title('Pokemon')

data_path = './data/pokemon/pokemon'

img_path = os.path.join(data_path, '3.png')

img = Image.open(img_path)
img = np.array(img)

# st.subheader('3.png')
# st.image(img, caption='Fushigibana', use_column_width=True)


genre = st.radio("What's", ('Comedy', 'Drame', 'Documentary'))

if genre == 'Comedy':
    st.text('You selected comdey')
else:
    st.write('fff')


G = models.Generator(z_dim=20, image_size=256)
D = models.Discriminator(z_dim=20, image_size=256)

input_z = torch.randn(1, 20, 1, 1)
fake_images = G(input_z)
d_out = D(fake_images)
d_out = nn.Sigmoid()(d_out)

print(d_out)
print(d_out.size())

# img_transform = fake_images[0].detach().permute(1, 2, 0).numpy()
#
# _max, _min = img_transform.max(), img_transform.min()
# img_transform = (img_transform - _min) * 255 / (_max - _min)
# img_transform = img_transform.astype(int)
#
#
# st.subheader('Generate')
# st.image(img_transform)
