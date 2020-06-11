
import glob

from torch.utils.data import DataLoader
from src import utils, models, trainer

mean = (0.5,)
std = (0.5,)
num_epochs = 20000
z_dim = 80
batch_size = 16
img_size = 64
img_path = glob.glob('./data/pokemon/*.png')

transform = utils.ImageTransform(img_size, mean, std)
dataset = utils.PokemonDataset(img_path=img_path, transform=transform)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

G = models.Generator_64(z_dim=z_dim, image_size=64)
D = models.Discriminator_64(image_size=64)

G.apply(utils.weights_init)
D.apply(utils.weights_init)

G, D = trainer.train_model(G, D, dataloader, z_dim=z_dim, num_epochs=num_epochs, save_weights_path='./weights',
                           tensorboard_path='./tensorboard', exp='GAN_01')

