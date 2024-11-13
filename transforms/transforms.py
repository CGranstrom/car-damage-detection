from torchvision import transforms

from utils.constants import IMAGENET_MEAN, IMAGENET_RES, IMAGENET_STD

train_transform = transforms.Compose(
    [
        transforms.Resize(IMAGENET_RES),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(IMAGENET_RES),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)
