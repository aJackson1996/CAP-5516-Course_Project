import math
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from DiffAugment_pytorch import rand_cutout, rand_translation

def categorize(dementia_rating):
    if 0 <= dementia_rating < .5:
        return torch.tensor(0)
    elif .5 <= dementia_rating < 1:
        return torch.tensor(1)
    elif 1 <= dementia_rating < 2:
        return torch.tensor(2)
    else:
        return torch.tensor(3)


def batch_data(dataset, batch_size):
    tensorize = transforms.ToTensor()
    batch_count = math.ceil(len(dataset) / batch_size)
    batched_data = []
    for i in range(batch_count):
        sample = dataset[i * batch_size: min(batch_size * (i + 1), len(dataset))]
        batched_data.append(torch.stack(sample))

    return batched_data


def generator_loss_non_saturated(discriminator_output):
    return -torch.mean(torch.log(discriminator_output + 1e-8))

#using this loss made the discriminator too good, so I swapped to BCELoss
def discriminator_loss_function(discriminator_output_real, discriminator_output_fake):
    return -torch.mean(torch.log(discriminator_output_real+1e-8) + torch.log(1 - (discriminator_output_fake + 1e-8)))


def augment_image(x):
    x = rand_cutout(x)
    x = rand_translation(x)
    flip_transform = transforms.RandomHorizontalFlip()
    x = flip_transform(x)
    return x

def get_images_from_path(img_path):
    MRI_images = []
    for root, subdirs, files in os.walk(img_path):
        for file in files:
            img_path = os.path.join(root, file)
            img = Image.open(img_path)
            img_transform = transforms.ToTensor()
            img_tensor = img_transform(img)
            MRI_images.append(img_tensor)
    return MRI_images