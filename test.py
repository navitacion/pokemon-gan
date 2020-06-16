import matplotlib.pyplot as plt
import torch
from src import models


net = models.Discriminator_dcgan(image_size=64, in_channel=3)

z = torch.randn(1, 3, 128, 128)

out = net(z)

print(out.size())

# plt.imshow(out[0].permute(1, 2, 0).detach().numpy())
# plt.show()
