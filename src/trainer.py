import os
import time
from tqdm import tqdm
import torch
from torch import nn

from tensorboardX import SummaryWriter


def train_model(G, D, dataloader, z_dim, num_epochs, save_weights_path, exp='DCGAN',
                tensorboard_path='./tensorboard', gan_type='DCGAN', save_weight_epoch=100):

    assert gan_type in ['DCGAN', 'SAGAN'], "This Trainer is supported 'DCGAN, SAGAN'"
    print(f'Pokemon {gan_type} Training...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    if not os.path.exists(tensorboard_path):
        os.mkdir(tensorboard_path)

    # Tensorboard
    writer = SummaryWriter(os.path.join(tensorboard_path, exp))

    # 最適化手法の設定
    g_lr, d_lr = 0.0002, 0.0002
    beta1, beta2 = 0.0, 0.9
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, (beta1, beta2))
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, (beta1, beta2))

    # 誤差関数を定義
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # ネットワークをGPUへ
    G.to(device)
    D.to(device)

    G.train()  # モデルを訓練モードに
    D.train()  # モデルを訓練モードに

    # 画像の枚数
    batch_size = dataloader.batch_size

    # イテレーションカウンタをセット
    iteration = 1

    # epochのループ
    for epoch in tqdm(range(num_epochs)):

        # 開始時刻を保存
        epoch_g_loss = 0.0  # epochの損失和
        epoch_d_loss = 0.0  # epochの損失和

        # データローダーからminibatchずつ取り出すループ
        for imges in dataloader:

            # --------------------
            # 1. Train Discriminator
            # --------------------
            # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
            if imges.size()[0] == 1:
                continue

            # GPUが使えるならGPUにデータを送る
            imges = imges.to(device)

            # 正解ラベルと偽ラベルを作成
            # epochの最後のイテレーションはミニバッチの数が少なくなる
            mini_batch_size = imges.size()[0]

            # 真の画像を判定
            d_out_real = D(imges)

            # 偽の画像を生成して判定
            input_z = torch.randn(mini_batch_size, z_dim, 1, 1).to(device)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            # Loss
            d_loss_real, d_loss_fake = 0, 0

            # BCEWithLogitsLoss
            if gan_type == 'DCGAN':
                label_real = torch.full((mini_batch_size,), 1, dtype=torch.float).to(device)
                label_fake = torch.full((mini_batch_size,), 0, dtype=torch.float).to(device)
                d_loss_real = criterion(d_out_real.view(-1), label_real)
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

            # Loss
            g_loss = 0
            # BCEWithLogitsLoss
            if gan_type == 'DCGAN':
                label_real = torch.full((mini_batch_size,), 1, dtype=torch.float).to(device)
                g_loss = criterion(d_out_fake.view(-1), label_real)

            # hinge version of the adversarial loss
            elif gan_type == 'SAGAN':
                g_loss = - d_out_fake.mean()

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # --------------------
            # 3. 記録
            # --------------------
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1

        D_loss = epoch_d_loss / batch_size
        G_loss = epoch_g_loss / batch_size

        writer.add_scalar('loss/netD', D_loss, epoch)
        writer.add_scalar('loss/netG', G_loss, epoch)

        if epoch % save_weight_epoch == 0:
            torch.save(G.state_dict(), os.path.join(save_weights_path, f'{exp}_netG_epoch_{epoch}.pth'))
            torch.save(D.state_dict(), os.path.join(save_weights_path, f'{exp}_netD_epoch_{epoch}.pth'))

    return G, D
