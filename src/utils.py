from PIL import Image

from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms


class PokemonDataset(Dataset):

    def __init__(self, img_path, transform):
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        target = self.img_path[idx]

        img = Image.open(target)
        img_transformed = self.transform(img)

        # 透過度は削除
        if img_transformed.size(0) == 4:
            img_transformed = img_transformed[:3, :, :]

        return img_transformed


class ImageTransform:
    def __init__(self, img_size, mean, std):
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.transform(img)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Conv2dとConvTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm2dの初期化
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
