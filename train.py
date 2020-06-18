import glob
import argparse
from torch.utils.data import DataLoader
from src import utils, models, trainer


# Parser  ################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('-img_s', '--image_size', type=int, default=128)
parser.add_argument('-epoch', '--epoch', type=int, default=1001)
parser.add_argument('-gan', '--gan_type', choices=['DCGAN', 'SAGAN'], default='SAGAN')
parser.add_argument('-s_epoch', '--save_weight_epoch', type=int, default=40)

args = parser.parse_args()

# グローバル変数  ################################################################
MEAN = (0.5,)
STD = (0.5,)
EPOCHS = args.epoch
Z_DIM = 400
BATCHSIZE = args.batch_size
IMGSIZE = args.image_size

# データの読み込み  ################################################################
img_path = glob.glob('./data/pokemon/*.png')
transform = utils.ImageTransform(IMGSIZE, MEAN, STD)
dataset = utils.PokemonDataset(img_path=img_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True)

# モデルの構築  ################################################################
if args.gan_type == 'DCGAN':
    G = models.Generator_dcgan(z_dim=Z_DIM, image_size=64, out_channel=3)
    D = models.Discriminator_dcgan(image_size=64, in_channel=3)
elif args.gan_type == 'SAGAN':
    G = models.Generator_sagan(z_dim=Z_DIM, image_size=48, out_channel=3)
    D = models.Discriminator_sagan(image_size=48, in_channel=3)

G.apply(utils.weights_init)
D.apply(utils.weights_init)

# 学習  ################################################################
trainer.train_model(G, D, dataloader, z_dim=Z_DIM, num_epochs=EPOCHS, save_weights_path='./weights',
                    tensorboard_path='./tensorboard', gan_type=args.gan_type, save_weight_epoch=args.save_weight_epoch)

