import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision
import torch
from sklearn.model_selection import KFold
from torch import nn, optim

from utils import get_images_from_path, categorize, batch_data


def train(model, data, labels, loss_func, optim, device, path):
    model.train()
    batched_data = batch_data(data, 8)
    batched_labels = batch_data(labels, 8)
    accuracies = []
    for epoch in range(1, 101):
        correct = 0
        num_samples = 0
        for i in range(len(batched_data)):
            images = batched_data[i].to(device)
            num_samples += images.shape[0]
            images = images.repeat(1, 3, 1, 1)
            label = torch.tensor(batched_labels[i])
            label = label.to(device)
            optim.zero_grad()
            output = model(images)
            output = torch.softmax(output, 1)
            pred = output.argmax(dim = 1, keepdim = True)
            loss = loss_func(output, label)
            matches = torch.eq(pred, label.reshape(label.shape[0],-1))
            correct += matches.sum().item()
            loss.backward()
            optim.step()
        accuracy = ((correct / num_samples) * 100)
        accuracies.append(accuracy)
        print(f"Epoch {epoch} done. Accuracy :{accuracy:.2f}%")

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.plot(range(1, epoch + 1), accuracies, "b", label='Training curve')
    ax1.legend(loc='lower left')
    plt.show()

    torch.save(
        model.state_dict(),
        os.path.join(weights_path, path),
    )

def eval(model, data, labels, device):
    model = model.to(device)
    model.eval()
    batched_data = batch_data(data, 8)
    batched_labels = batch_data(labels, 8)

    with torch.no_grad():
        correct = 0
        num_samples = 0
        tp_mci = 0
        tp_ad = 0
        fn_mci = 0
        fn_ad = 0
        for i in range(len(batched_data)):
            images = batched_data[i].to(device)
            num_samples += images.shape[0]
            images = images.repeat(1, 3, 1, 1)
            label = torch.tensor(batched_labels[i])
            label = label.to(device)
            output = model(images)
            output = torch.softmax(output, 1)
            pred = output.argmax(dim = 1, keepdim = True)
            matches = torch.eq(pred, label.reshape(label.shape[0],-1))
            correct += matches.sum().item()
            for idx in range(len(label)):
                if label[idx] == 1:
                    if pred[idx] == 1:
                        tp_mci += 1
                    else:
                        fn_mci += 1
                if label[idx] == 2:
                    if pred[idx] == 2:
                        tp_ad += 1
                    else:
                        fn_ad += 1
        recall_mci = tp_mci / (tp_mci + fn_mci) * 100
        recall_ad = tp_ad / (tp_ad + fn_ad) * 100
        print(f"Evaluation done. Accuracy :{((correct / num_samples) * 100):.2f}%. Recall for MCI:{recall_mci:.2f}%. Recall for AD:{recall_ad:.2f}%")

parser = argparse.ArgumentParser('Classifier.')
parser.add_argument('--Train_Classifiers',
                    type=bool, default=False,
                    help='Set to true to train 5 different classifiers using the base data set unioned with varying subsets of generated images.')
parser.add_argument('--Evaluate_Classifiers',
                    type=bool, default=True,
                    help='Set to true to evaluate the 5 models.')
# load face detector
FLAGS = None
FLAGS, unparsed = parser.parse_known_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_function = nn.CrossEntropyLoss()

base_image_path = os.path.join(os.path.dirname(__file__), 'oasis-mri')
weights_path = os.path.join(os.path.dirname(__file__), 'classifier_weights')
session_csv = pd.read_csv(os.path.join(base_image_path, 'oasis_cross-sectional.csv'))
session_csv['ID'] = session_csv['ID'].str.split('_')
session_csv['ID'] = session_csv['ID'].str[1]
session_csv.set_index('ID', inplace=True)
latent_dim_size = 100
session_csv['MRI_image'] = [[] for i in range(len(session_csv))]
training_data = []
labels = []


base_image_path = os.path.join(os.path.dirname(__file__), 'oasis-mri')
base_images = get_images_from_path(base_image_path)

resize_transform = torchvision.transforms.Resize((224, 224))
base_images = [resize_transform(x) for x in base_images]

session_csv['MRI_image'] = base_images

for idx, row in session_csv.iterrows():
    CDR = row.loc['CDR']
    training_data.append(row['MRI_image'])
    labels.append(torch.tensor(CDR))

#labels = [torch.tensor(i) for i in session_csv['CDR'].values]
labels = [categorize(i) for i in labels]


kf = KFold(n_splits = 5)

training_indices = []
testing_indices = []

for fold_index, (train_indices, test_indices) in enumerate(kf.split(training_data)):
    training_indices.append(train_indices)
    testing_indices.append(test_indices)


train_split_data = []
train_split_labels = []

test_split_data = []
test_split_labels = []

for training_index in training_indices[0]:
    train_split_data.append(training_data[training_index])
    train_split_labels.append(labels[training_index])

for testing_index in testing_indices[0]:
    test_split_data.append(training_data[testing_index])
    test_split_labels.append(labels[testing_index])

if FLAGS.Train_Classifiers:
    #train a model using the base dataset and save the params
    model_base = torchvision.models.resnet18(pretrained=True)
    num_features = model_base.fc.in_features
    model_base.fc = nn.Linear(num_features, 4)  #overwrite the fc layer to output predictions for 4 classes
    model_base = model_base.to(device)
    optimizer_base = optim.SGD(model_base.parameters(), lr=0.001)
    #train(model_base, train_split_data, train_split_labels, loss_function, optimizer_base, device, "best_weights_base.pth")

    model_healthy = torchvision.models.resnet18(pretrained=True)
    num_features = model_healthy.fc.in_features
    model_healthy.fc = nn.Linear(num_features, 4)
    model_healthy = model_healthy.to(device)
    optimizer_healthy = optim.SGD(model_healthy.parameters(), lr=0.001)

    healthy_images = get_images_from_path(os.path.join(os.path.dirname(__file__), 'train\generated_healthy'), for_CGAN = False)
    training_data_more_healthy = train_split_data
    labels_more_healthy = train_split_labels
    for image in healthy_images:
        image = resize_transform(image)
        training_data_more_healthy.append(image)
        labels_more_healthy.append(torch.tensor(0))

    #train(model_healthy, training_data_more_healthy, labels_more_healthy, loss_function, optimizer_healthy, device, "best_weights_added_healthy.pth")

    model_mci = torchvision.models.resnet18(pretrained=True)
    num_features = model_mci.fc.in_features
    model_mci.fc = nn.Linear(num_features, 4)  #overwrite the fc layer to output predictions for 4 classes
    model_mci = model_mci.to(device)
    optimizer_mci = optim.SGD(model_mci.parameters(), lr=0.001)

    mci_images = get_images_from_path(os.path.join(os.path.dirname(__file__), 'train\generated_mci'), for_CGAN = False)
    training_data_more_mci = train_split_data
    labels_more_mci = train_split_labels
    for image in mci_images:
        image = resize_transform(image)
        training_data_more_mci.append(image)
        labels_more_mci.append(torch.tensor(1))

    #train(model_mci, training_data_more_mci, labels_more_mci, loss_function, optimizer_mci, device, "best_weights_added_mci.pth")

    model_ad = torchvision.models.resnet18(pretrained=True)
    num_features = model_ad.fc.in_features
    model_ad.fc = nn.Linear(num_features, 4)
    model_ad = model_ad.to(device)
    optimizer_ad = optim.SGD(model_ad.parameters(), lr=0.001)

    ad_images = get_images_from_path(os.path.join(os.path.dirname(__file__), 'train\generated_ad'), for_CGAN = False)
    training_data_more_ad = train_split_data
    labels_more_ad = train_split_labels
    for image in ad_images:
        image = resize_transform(image)
        training_data_more_ad.append(image)
        labels_more_ad.append(torch.tensor(2))

    #train(model_ad, training_data_more_ad, labels_more_ad, loss_function, optimizer_ad, device, "best_weights_added_ad.pth")

    model_unhealthy = torchvision.models.resnet18(pretrained=True)
    num_features = model_unhealthy.fc.in_features
    model_unhealthy.fc = nn.Linear(num_features, 4)
    model_unhealthy = model_unhealthy.to(device)
    optimizer_unhealthy = optim.SGD(model_unhealthy.parameters(), lr=0.001)

    training_data_more_unhealthy = training_data_more_mci
    labels_more_unhealthy = labels_more_mci
    for image in ad_images:
        image = resize_transform(image)
        training_data_more_unhealthy.append(image)
        labels_more_ad.append(torch.tensor(2))

    train(model_unhealthy, training_data_more_unhealthy, labels_more_unhealthy, loss_function, optimizer_unhealthy, device, "best_weights_added_unhealthy.pth")

    model_all = torchvision.models.resnet18(pretrained=True)
    num_features = model_all.fc.in_features
    model_all.fc = nn.Linear(num_features, 4)
    model_all = model_all.to(device)
    optimizer_all = optim.SGD(model_all.parameters(), lr=0.001)

    training_data_more_all = training_data_more_unhealthy
    labels_more_all = labels_more_unhealthy
    for image in healthy_images:
        image = resize_transform(image)
        training_data_more_all.append(image)
        labels_more_all.append(torch.tensor(0))

    train(model_all, training_data_more_all, labels_more_all, loss_function, optimizer_all,
          device, "best_weights_added_all.pth")

if FLAGS.Evaluate_Classifiers:
    test_healthy_images = get_images_from_path(os.path.join(os.path.dirname(__file__), 'test\generated_healthy'),
                                          for_CGAN=False)
    test_mci_images = get_images_from_path(os.path.join(os.path.dirname(__file__), 'test\generated_mci'),
                                          for_CGAN=False)
    test_ad_images = get_images_from_path(os.path.join(os.path.dirname(__file__), 'test\generated_ad'),
                                          for_CGAN=False)

    for test_healthy_image in test_healthy_images:
        test_healthy_image = resize_transform(test_healthy_image)
        test_split_data.append(test_healthy_image)
        test_split_labels.append(torch.tensor(0))

    for test_mci_image in test_mci_images:
        test_mci_image = resize_transform(test_mci_image)
        test_split_data.append(test_mci_image)
        test_split_labels.append(torch.tensor(1))

    for test_ad_image in test_ad_images:
        test_ad_image = resize_transform(test_ad_image)
        test_split_data.append(test_ad_image)
        test_split_labels.append(torch.tensor(2))

    models = []
    model_trained_base = torchvision.models.resnet18(pretrained=True)
    num_features = model_trained_base.fc.in_features
    model_trained_base.fc = nn.Linear(num_features, 4)
    model_trained_base.load_state_dict(torch.load(os.path.join(weights_path, "best_weights_base.pth")))
    models.append(model_trained_base)

    model_trained_healthy = torchvision.models.resnet18(pretrained=True)
    model_trained_healthy.fc = nn.Linear(num_features, 4)
    model_trained_healthy.load_state_dict(torch.load(os.path.join(weights_path, "best_weights_added_healthy.pth")))
    models.append(model_trained_healthy)

    model_trained_mci = torchvision.models.resnet18(pretrained=True)
    model_trained_mci.fc = nn.Linear(num_features, 4)
    model_trained_mci.load_state_dict(torch.load(os.path.join(weights_path, "best_weights_added_mci.pth")))
    models.append(model_trained_mci)

    model_trained_ad = torchvision.models.resnet18(pretrained=True)
    model_trained_ad.fc = nn.Linear(num_features, 4)
    model_trained_ad.load_state_dict(torch.load(os.path.join(weights_path, "best_weights_added_ad.pth")))
    models.append(model_trained_ad)

    model_trained_unhealthy = torchvision.models.resnet18(pretrained=True)
    model_trained_unhealthy.fc = nn.Linear(num_features, 4)
    model_trained_unhealthy.load_state_dict(torch.load(os.path.join(weights_path, "best_weights_added_unhealthy.pth")))
    models.append(model_trained_unhealthy)

    model_trained_all = torchvision.models.resnet18(pretrained=True)
    model_trained_all.fc = nn.Linear(num_features, 4)
    model_trained_all.load_state_dict(torch.load(os.path.join(weights_path, "best_weights_added_all.pth")))
    models.append(model_trained_all)

    for model in models:
        eval(model, test_split_data, test_split_labels, device)
