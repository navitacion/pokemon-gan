import numpy as np
import matplotlib.pyplot as plt
import torch
import streamlit as st
import torchvision.utils as vutils

from src import models

# グローバル変数
Z_DIM = 400
OUTPUT_IMAGE_NUM = 20
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# モデルの初期化関数
@st.cache
def model_init(weight_path, exp):
    """
    モデルの初期化関数

    Parameters
    ----------
    weight_path : str
        学習済みモデルの重みのパス
    exp : str
        モデルの種類（DCGAN, SAGAN）

    Returns
    -------
    G : torch.nn.Module
        学習済みモデル
    """

    if 'DCGAN' in exp:
        G = models.Generator_dcgan(z_dim=Z_DIM, image_size=64, out_channel=3)
    elif 'SAGAN' in exp:
        G = models.Generator_sagan(z_dim=Z_DIM, image_size=48, out_channel=3)
    G.load_state_dict(torch.load(weight_path, map_location=device))
    G.eval()

    return G


# タイトルとディスクリプションの設定
st.title('Pokémon GAN')
st.markdown('---')
st.markdown('This app is a demo of GAN using Pokémon images.')
st.markdown('In this demo, the DCGAN and SAGAN models can be used to generate images.')
st.markdown('By selecting the type of model and the number of training epochs, '
            'you can see the results of the Generative Images from GAN.')
st.markdown('Maybe, you can find a new Pokémon...   Enjoy!!')
st.markdown('---')

# スライドバーの設定
st.sidebar.subheader('Setup')
exp = st.sidebar.selectbox('Select GAN', ('DCGAN', 'SAGAN'))
epoch = st.sidebar.slider('Select Epoch', min_value=0, max_value=1000, step=20)

# モデルの準備
weight_path = f'./weights/{exp}_netG_epoch_{epoch}.pth'
G = model_init(weight_path, exp)

# 画像生成
z = torch.randn(OUTPUT_IMAGE_NUM, Z_DIM, 1, 1)
with torch.no_grad():
    out = G(z)

# 複数画像を一つに結合
img = vutils.make_grid(out.detach().cpu(), normalize=True, padding=2, nrow=5, pad_value=1)
img = np.transpose(img.numpy(), [1, 2, 0])

# pyplotで図の構成を作成
plt.imshow(img)
plt.axis('off')
plt.tight_layout()

# アプリ上で画像を表示
st.subheader('Generative Pokémon Images')
st.pyplot()

