
import glob

from torch.utils.data import DataLoader
from src import utils, models, trainer

mean = (0.5,)
std = (0.5,)
num_epochs = 200
img_path = glob.glob('../data/pokemon/pokemon/*.png')
transform = utils.ImageTransform(mean, std)
dataset = utils.PokemonDataset(img_path=img_path, transform=transform)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

G = models.Generator(z_dim=20, image_size=256)
D = models.Discriminator(z_dim=20, image_size=256)

G.apply(utils.weights_init)
D.apply(utils.weights_init)

G, D = trainer.train_model(G, D, dataloader, num_epochs=num_epochs)

