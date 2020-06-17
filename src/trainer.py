import os
import time
from tqdm import tqdm
import torch
from torch import nn

from tensorboardX import SummaryWriter


# 学習メモ
# DCGAN: batch_size=64  Epoch 1000  2h20m
# SAGAN: batch_size=16  Epoch 1000


def train_model(G, D, dataloader, z_dim, num_epochs, save_weights_path='./weights',
                tensorboard_path='./tensorboard', gan_type='DCGAN', save_weight_epoch=100):
    """
    モデル学習のヘルパー関数

    Parameters
    ----------
    G : torch.nn.Modulue
        生成モデル
    D : torch.nn.Module
        識別モデル
    dataloader: torch.utils.data.DataLoader
        データローダー
    num_epochs: int
        最大エポック数
    save_weights_path: str
        学習モデルの重み保存先ディレクトリ
    tensorboard_path: str
        tensorboardのログ出力先ディレクトリ
    gan_type: str
        学習するGANのタイプ（DCGAN, SAGAN）
    save_weight_epoch: int
        重みを保存するタイミング
    """

    assert gan_type in ['DCGAN', 'SAGAN'], "This Trainer is supported 'DCGAN, SAGAN'"
    print(f'Pokemon {gan_type} Training...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    if not os.path.exists(tensorboard_path):
        os.mkdir(tensorboard_path)

    # Tensorboard
    writer = SummaryWriter(os.path.join(tensorboard_path, gan_type))

    # 最適化手法の設定
    g_lr, d_lr = 0, 0
    if gan_type == 'DCGAN':
        g_lr, d_lr = 0.0002, 0.0002
    elif gan_type == 'SAGAN':
        g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, (beta1, beta2))
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, (beta1, beta2))

    # 誤差関数を定義
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # ネットワークをGPUへ
    G.to(device)
    D.to(device)

    G.train()
    D.train()

    # 画像の枚数
    batch_size = dataloader.batch_size

    # イテレーションカウンタをセット
    iteration = 1

    # epochのループ
    for epoch in tqdm(range(num_epochs)):

        # Epoch Lossの初期化
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for imges in dataloader:

            # --------------------
            # 1. Train Discriminator
            # --------------------
            # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
            if imges.size()[0] == 1:
                continue

            imges = imges.to(device)
            mini_batch_size = imges.size()[0]

            # 真の画像を判定
            d_out_real = D(imges)

            # 偽の画像を生成して判定
            input_z = torch.randn(mini_batch_size, z_dim, 1, 1).to(device)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            # Batch Lossの初期化
            d_loss_real, d_loss_fake = 0, 0

            # BCEWithLogitsLoss
            if gan_type == 'DCGAN':
                label_real = torch.full((mini_batch_size,), 1, dtype=torch.float).to(device)
                label_fake = torch.full((mini_batch_size,), 0, dtype=torch.float).to(device)
                # 真の画像を真と判定する
                d_loss_real = criterion(d_out_real.view(-1), label_real)
                # 偽の画像を偽と判定する
                d_loss_fake = criterion(d_out_fake.view(-1), label_fake)

            # hinge version of the adversarial loss
            elif gan_type == 'SAGAN':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            d_loss = d_loss_real + d_loss_fake

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            # --------------------
            # 2. Generatorの学習
            # --------------------
            # 偽の画像を生成して判定
            input_z = torch.randn(mini_batch_size, z_dim, 1, 1).to(device)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            # Batch Lossの初期化
            g_loss = 0
            # BCEWithLogitsLoss
            if gan_type == 'DCGAN':
                label_real = torch.full((mini_batch_size,), 1, dtype=torch.float).to(device)
                # 偽の画像を真と判定する
                g_loss = criterion(d_out_fake.view(-1), label_real)

            # hinge version of the adversarial loss
            elif gan_type == 'SAGAN':
                g_loss = - d_out_fake.mean()

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Epoch Lossの更新
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            # イテレーションカウンタの更新
            iteration += 1

        # Epoch Lossの平均を算出
        D_loss = epoch_d_loss / batch_size
        G_loss = epoch_g_loss / batch_size

        # Tensorboardへの書き出し
        writer.add_scalar('loss/netD', D_loss, epoch)
        writer.add_scalar('loss/netG', G_loss, epoch)

        # save_weight_epochで定義したタイミングでモデルの重みを保存
        if epoch % save_weight_epoch == 0:
            torch.save(G.state_dict(), os.path.join(save_weights_path, f'{gan_type}_netG_epoch_{epoch}.pth'))
            torch.save(D.state_dict(), os.path.join(save_weights_path, f'{gan_type}_netD_epoch_{epoch}.pth'))
