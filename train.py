
import glob
import torch
from torch.utils.data import DataLoader
from src import utils, models, trainer

mean = (0.5,)
std = (0.5,)
num_epochs = 5001
z_dim = 500
batch_size = 32
img_size = 64
img_path = glob.glob('./data/pokemon/*.png')

transform = utils.ImageTransform(img_size, mean, std)
dataset = utils.PokemonDataset(img_path=img_path, transform=transform)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

G = models.Generator_sagan(z_dim=z_dim, image_size=64)
D = models.Discriminator_sagan(image_size=64)

# G.load_state_dict(torch.load('./weights/GAN_01_netG_epoch_500.pth', map_location=torch.device('cuda:0')))
# D.load_state_dict(torch.load('./weights/GAN_01_netD_epoch_500.pth', map_location=torch.device('cuda:0')))

G.apply(utils.weights_init)
D.apply(utils.weights_init)

G, D = trainer.train_model_sagan(G, D, dataloader, z_dim=z_dim, num_epochs=num_epochs, save_weights_path='./weights',
                                 tensorboard_path='./tensorboard', exp='SAGAN_01')

