import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super(Discriminator, self).__init__()

        self.embedded_discriminator_label = nn.Sequential(
            nn.Embedding(num_classes, embedding_dim),
            nn.Linear(embedding_dim, 208*176)
        )

        self.Model = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1, bias=False), #feature map dimensions after layer: 208 x 176
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(32, 64, 5, 2, 2, bias=False), #feature map dimensions after layer: 104 x 88
            nn.BatchNorm2d(64,momentum=.1, eps=.8),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(64,128 , 5, 2, 2, bias=False), #feature map dimensions after layer: 52 x 44
            nn.BatchNorm2d(128, momentum=.1, eps=.8),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False), #feature map dimensions after layer: 26 x 22
            nn.BatchNorm2d(256, momentum=.1, eps=.8),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),  # feature map dimensions after layer: 13 x 11
            nn.BatchNorm2d(512, momentum=.1, eps=.8),
            nn.LeakyReLU(.2, inplace=True),
            nn.Flatten(),
            nn.Linear(512 * 13 * 11, 1),
            nn.Sigmoid()
        )


    def forward(self, X):
        img, label = X
        embedded_label = self.embedded_discriminator_label(label)
        embedded_label = embedded_label.reshape(-1, 1, 208, 176)
        discriminator_input = torch.concat((img, embedded_label), dim=1)
        return self.Model(discriminator_input)

class Generator(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super(Generator, self).__init__()

        self.embedded_generator_label = nn.Sequential(
            nn.Embedding(num_classes, embedding_dim),
            nn.Linear(embedding_dim, 11*13)
        )

        self.latent_space_generation = nn.Sequential(
            nn.Linear(100, 256 * 11 * 13), #The idea here is that we are using the latent vector (dim 1, 100) to generate 256 14 x 14 copies
            #You need these copies to learn some large number of low dimensional image mappings from the latent space
            nn.LeakyReLU(.2, inplace=True)
        )

        self.Model = nn.Sequential(
            nn.ConvTranspose2d(257, 1024, 4, 2, 1, bias=False),  # feature map dimensions after layer: 26 x 22
            nn.BatchNorm2d(1024, momentum=.1, eps=.8),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False), #feature map dimensions after layer: 52 x 44
            nn.BatchNorm2d(512, momentum=.1, eps=.8),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), #feature map dimensions after layer: 104 x 88
            nn.BatchNorm2d(256, momentum=.1, eps=.8),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 1, 1, bias=False),  # feature map dimensions after layer: 104 x 88
            nn.BatchNorm2d(128, momentum=.1, eps=.8),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False), #feature map dimensions after layer: 208 x 176
            #nn.BatchNorm2d(64, momentum=.1, eps=.8),
            #nn.ReLU(True),
            #nn.ConvTranspose2d(64, 1, 3, 1, 1, bias=False),  # feature map dimensions after layer: 208 x 176
            nn.Sigmoid()
        )

    def forward(self, inputs):
        noise_vector, label = inputs
        embedded_label = self.embedded_generator_label(label)
        embedded_label = embedded_label.reshape(-1, 1, 13, 11)
        latent_space_input = self.latent_space_generation(noise_vector) #generate a latent space representation from the noise input
        latent_space_input = latent_space_input.reshape(-1, 256, 13, 11)
        generator_input = torch.concat((embedded_label, latent_space_input), dim=1)
        generated_image = self.Model(generator_input)
        return generated_image