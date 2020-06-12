import matplotlib.pyplot as plt
import torch
import streamlit as st

from src import models


Z_DIM = 500

@st.cache
def model_init(weight_path, exp):
    if 'DCGAN' in exp:
        G = models.Generator_dcgan(z_dim=Z_DIM, image_size=64)
    elif 'SAGAN' in exp:
        G = models.Generator_sagan(z_dim=Z_DIM, image_size=64)
    G.load_state_dict(torch.load(weight_path, map_location=device))
    return G


st.title('Pokemon GAN')

exp = st.selectbox('Select GAN', ('DCGAN_01', 'SAGAN_01'))

epoch = st.slider('Select Epoch', min_value=0, max_value=5000, step=500)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

weight_path = f'./weights/{exp}_netG_epoch_{epoch}.pth'
G = model_init(weight_path, exp)

fig, axes = plt.subplots(ncols=5, nrows=3, figsize=(16, 12))
for i, ax in enumerate(axes.ravel()):
    input_z = torch.randn(1, Z_DIM, 1, 1)
    d_out = G(input_z)  # (1, image_size, image_size, 1)
    out = d_out[0].squeeze()  # (image_size, image_size)
    out = out.detach().numpy()

    # Reverse Normalize
    _max, _min = out.max(), out.min()
    out = (out - _min) * 255 / (_max - _min)
    out = out.astype(int)

    ax.imshow(out, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')

plt.tight_layout()

st.subheader('Generative Pokemon Images')
st.pyplot()
