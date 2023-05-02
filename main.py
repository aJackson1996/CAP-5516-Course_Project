# This is a sample Python script.
import argparse
import math
import os.path
from random import random, randint

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.models
from PIL import Image
from matplotlib import pyplot as plt
from torch import optim
from torchvision import transforms
import torch.nn.functional as F
from DiffAugment_pytorch import rand_cutout, rand_translation
import models
from utils import augment_image, categorize, batch_data, generator_loss_non_saturated, get_images_from_path


def train(train_set, train_labels, k, smoothing_factor, discriminator, discriminator_loss, discriminator_optimizer,
          generator, generator_loss, generator_optimizer, device, latent_dim):

    gen_losses = []
    discriminator_losses = []
    best_fake_loss = 0
    best_fake_images = []
    best_real_images = []
    discriminator.train()
    generator.train()
    for i in range(len(train_set)):
        real_images = train_set[i]
        labels = train_labels[i]
        real_images, labels = real_images.to(device), labels.to(device)
        labels = labels.unsqueeze(1).long()
        augmented_real_images = augment_image(real_images)
        # Train Discriminator k times to every 1 time the generator is trained

        # Real data

        discriminator_vs_real = discriminator((augmented_real_images, labels))
        # array of ones tells the discriminator that all of these images should be predicted as real
        real_image_labels = (torch.ones((real_images.size(0)), 1) - smoothing_factor).to(device)
        # Array of zeroes tells discriminator these generated images should be predicted as fake
        fake_image_labels = torch.zeros(real_images.size(0), 1).to(device)

        d_real_loss = discriminator_loss(discriminator_vs_real, real_image_labels)

        # Fake data

        # generate real_images.size(0) images from noise vectors of size latent_dim
        noise_vector = torch.randn(real_images.size(0), latent_dim, device=device)
        noise_vector = noise_vector.to(device)  # transfer this batch of noise vectors to device
        generated_images = generator((noise_vector, labels))
        augmented_generated_images = augment_image(generated_images.detach())
        discriminator_vs_fake_discriminatorloss = discriminator((augmented_generated_images, labels))
        d_fake_loss = discriminator_loss(discriminator_vs_fake_discriminatorloss, fake_image_labels)
        d_loss = (d_real_loss + d_fake_loss) / 2
        discriminator_losses.append(d_loss.item())

        discriminator_optimizer.zero_grad()
        d_loss.backward()  # generate values to update weights with
        discriminator_optimizer.step()  # propagate generated values backwards
        if(d_loss > best_fake_loss):
            best_fake_loss = d_loss
            best_fake_images = generated_images
            best_real_images = real_images
        if ((i - 1) % k) == 0:  # After training discriminator k times, train generator once
            generator_optimizer.zero_grad()
            # array of ones makes it so that the generator is aiming to get the discriminator to predict its fake
            # images as real we do this again instead of reusing the discriminator output from above because we don't
            # want to do detach() which prevents the weights from propagating
            discriminator_vs_fake_generatorloss = discriminator((augment_image(generated_images), labels))
            #g_loss = generator_loss(discriminator_vs_fake_generatorloss, real_image_labels)
            g_loss = generator_loss_non_saturated(discriminator_vs_fake_generatorloss)
            gen_losses.append(g_loss.item())
            g_loss.backward()
            generator_optimizer.step()
    avg_gen_loss = float(np.mean(gen_losses))
    avg_disc_loss = float(np.mean(discriminator_losses))
    print("Avg Generator Loss: {}, Avg Discriminator Loss: {}".format(avg_gen_loss, avg_disc_loss))
    return best_fake_images, best_real_images, avg_gen_loss, avg_disc_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser('C-GAN MRI.')
    parser.add_argument('--train_generator',
                        type=bool, default=False,
                        help='Set to true if you need to generate parameters for the model.')
    parser.add_argument('--generate_images',
                        type=bool, default=True,
                        help='Set to true if you need to generate images using the model.')
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    absolute_path = os.path.join(os.path.dirname(__file__), 'oasis-mri')
    params_path = os.path.join(os.path.dirname(__file__), "trained_generator_params.pth")
    absolute_path = '/'.join(absolute_path.split("\\"))
    session_csv = pd.read_csv(os.path.join(absolute_path, 'oasis_cross-sectional.csv'))

    session_csv['ID'] = session_csv['ID'].str.split('_')
    session_csv['ID'] = session_csv['ID'].str[1]
    session_csv.set_index('ID', inplace=True)
    latent_dim_size = 100
    session_csv['MRI_image'] = [[] for i in range(len(session_csv))]
    training_data = []
    labels = []
    MRI_images = get_images_from_path(absolute_path)

    stacked_images = torch.stack(MRI_images)
    def normalize(x : torch.Tensor):
        min = torch.min(x)
        max = torch.max(x)
        x -= min
        x /= max
        return x
    MRI_images = [normalize(x) for x in MRI_images]
    session_csv['MRI_image'] = MRI_images

    for idx, data in session_csv.iterrows():
        training_data.append(data.loc['MRI_image'])
        CDR = data.loc['CDR']
        labels.append(torch.tensor(CDR))

    labels = [categorize(i) for i in labels]

    batched_data = batch_data(training_data, 8)
    batched_labels = batch_data(labels, 8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor_to_image = transforms.ToPILImage()
    if FLAGS.train_generator:
        discriminator_model = models.Discriminator(4, 100).to(device)
        generator_model = models.Generator(4, 100).to(device)

        BCE_discriminator = torch.nn.BCELoss()
        BCE_generator = torch.nn.BCELoss()
        discriminator_optimizer = optim.Adam(discriminator_model.parameters(), lr=.0002)
        generator_optimizer = optim.Adam(generator_model.parameters(), lr=.0002)
        epoch_gen_losses = []
        epoch_disc_losses = []
        for epoch in range(1, 1501):
            images, real_images, gen_loss_for_epoch, disc_loss_for_epoch = train(batched_data, batched_labels, 1, 0, discriminator_model, BCE_discriminator, discriminator_optimizer,
                generator_model, BCE_generator, generator_optimizer, device, latent_dim_size)
            epoch_gen_losses.append(gen_loss_for_epoch)
            epoch_disc_losses.append(disc_loss_for_epoch)
            if epoch % 500 == 0:
                for i in range(len(images)):
                    fake_image = tensor_to_image(images[i])
                    plt.imshow(fake_image, cmap = plt.cm.gray)
                    plt.show()
                    plt.close()

        torch.save(
            generator_model.state_dict(),
            params_path,
        )
        print("saved trained generator model")

        fig2 = plt.figure(2)
        ax = fig2.add_subplot(111)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.plot(range(1, 1501), epoch_disc_losses, "b", label='Discriminator Loss Curve')
        ax.legend(loc='lower left')
        plt.show(block=True)
        plt.interactive(False)
    if FLAGS.generate_images:
        model = models.Generator(4, 100).to(device)
        model.load_state_dict(torch.load(params_path))
        #generated 100 images of each type for training, and 50 images of each type for testing
        #class 3 is left out since that is a placeholder class for items where the CDR was not specified
        for label in range(3):
            for image_idx in range(100):
                tensorized_label = torch.tensor(label).unsqueeze(0).long().to(device)
                noise_vector = torch.rand(1, latent_dim_size).to(device)
                generated_image = model((noise_vector, tensorized_label))
                generated_image = tensor_to_image(generated_image[0])
                if (label == 0):
                    file_path = os.path.join(os.path.dirname(__file__), 'train\generated_healthy')
                    generated_image = generated_image.save(os.path.join(file_path, f"healthy_{image_idx}.png"))
                if (label == 1):
                    file_path = os.path.join(os.path.dirname(__file__), 'train\generated_mci')
                    generated_image = generated_image.save(os.path.join(file_path, f"mci_{image_idx}.png"))
                if (label == 2):
                    file_path = os.path.join(os.path.dirname(__file__), 'train\generated_ad')
                    generated_image = generated_image.save(os.path.join(file_path, f"ad_{image_idx}.png"))
            for image_idx in range(50):
                tensorized_label = torch.tensor(label).unsqueeze(0).long().to(device)
                noise_vector = torch.rand(1, latent_dim_size).to(device)
                generated_image = model((noise_vector, tensorized_label))
                generated_image = tensor_to_image(generated_image[0])
                if(label == 0):
                    file_path = os.path.join(os.path.dirname(__file__), 'test\generated_healthy')
                    generated_image = generated_image.save(os.path.join(file_path, f"healthy_{image_idx}.png"))
                if (label == 1):
                    file_path = os.path.join(os.path.dirname(__file__), 'test\generated_mci')
                    generated_image = generated_image.save(os.path.join(file_path, f"mci_{image_idx}.png"))
                if (label == 2):
                    file_path = os.path.join(os.path.dirname(__file__), 'test\generated_ad')
                    generated_image = generated_image.save(os.path.join(file_path, f"ad_{image_idx}.png"))
