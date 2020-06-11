import glob
import argparse
from torch.utils.data import DataLoader
from src import utils, models, trainer


# Parser  ################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('-img_s', '--image_size', type=int, default=64)
parser.add_argument('-epoch', '--epoch', type=int, default=5001)
parser.add_argument('-gan', '--gan_type', choices=['DCGAN', 'SAGAN'], default='SAGAN')
parser.add_argument('-exp', '--exp_name')

args = parser.parse_args()

# Global  ################################################################
MEAN = (0.5,)
STD = (0.5,)
EPOCHS = args.epoch
Z_DIM = 500
BATCHSIZE = args.batch_size
IMGSIZE = args.image_size

# Data Loading  ################################################################
img_path = glob.glob('./data/pokemon/*.png')
transform = utils.ImageTransform(IMGSIZE, MEAN, STD)
dataset = utils.PokemonDataset(img_path=img_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True)

# Model  ################################################################

if args.gan_type == 'DCGAN':
    G = models.Generator_dcgan(z_dim=Z_DIM, image_size=64)
    D = models.Discriminator_dcgan(image_size=64)
elif args.gan_type == 'SAGAN':
    G = models.Generator_sagan(z_dim=Z_DIM, image_size=64)
    D = models.Discriminator_sagan(image_size=64)

G.apply(utils.weights_init)
D.apply(utils.weights_init)

# Training  ################################################################
G, D = trainer.train_model(G, D, dataloader, z_dim=Z_DIM, num_epochs=EPOCHS, save_weights_path='./weights',
                           tensorboard_path='./tensorboard', exp=args.exp_name, gan_type=args.gan_type)

