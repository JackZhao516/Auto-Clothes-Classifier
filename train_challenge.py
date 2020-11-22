'''
EECS 445 - Introduction to Machine Learning
Fall 2020 - Project 2
Train Challenge
    Trains a neural network to classify images
    Periodically outputs training information, and saves model checkpoints
    Usage: python train_challenge.py
'''
import torch
import numpy as np
import utils
from dataset import get_train_val_test_loaders
from model.challenge import Challenge
from train_common import *
from utils import config
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models
# resnet18 = models.resnet18()
# alexnet = models.alexnet()
# vgg16 = models.vgg16()
# squeezenet = models.squeezenet1_0()
# densenet = models.densenet161()
# inception = models.inception_v3()
# googlenet = models.googlenet()
# shufflenet = models.shufflenet_v2_x1_0()
# mobilenet = models.mobilenet_v2()
# resnext50_32x4d = models.resnext50_32x4d()
# wide_resnet50_2 = models.wide_resnet50_2()
# mnasnet = models.mnasnet1_0()


# GPU
device = torch.device("cuda")

def _train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    # TODO: complete the training step
    for i, (X, y) in enumerate(data_loader):
        # clear parameter gradients
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    #

def _evaluate_epoch(axes, tr_loader, val_loader, model, criterion, epoch, stats):
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in tr_loader:
            X,y = X.to(device), y.to(device)
            output = model(X)
            predicted = torch.argmax(output.data, dim=1)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
        train_loss = np.mean(running_loss)
        train_acc = correct / total
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in val_loader:
            X,y = X.to(device), y.to(device)
            output = model(X)
            predicted = torch.argmax(output.data, dim=1)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
        val_loss = np.mean(running_loss)
        val_acc = correct / total
    stats.append([val_acc, val_loss, train_acc, train_loss])
    utils.log_cnn_training(epoch, stats)
    utils.update_cnn_training_plot(axes, epoch, stats)

def main():
    # data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        num_classes=config('challenge.num_classes'))

    # TODO: define model, loss function, and optimizer
    # Model

    model = Challenge()
    model.to(device)
    # TODO: define loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    #

    # Attempts to restore the latest checkpoint if exists
    print('Loading challenge...')
    model, start_epoch, stats = restore_checkpoint(model,
        config('challenge.checkpoint'))

    fig, axes = utils.make_cnn_training_plot(name='Challenge')

    # Evaluate model
    _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, start_epoch,
        stats)

    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config('challenge.num_epochs')):
        # Train model
        _train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, epoch+1,
            stats)

        # Save model parameters
        save_checkpoint(model, epoch+1, config('challenge.checkpoint'), stats)

    print('Finished Training')

    # Keep plot open
    utils.save_cnn_training_plot(fig, name='challenge')
    utils.hold_training_plot()

if __name__ == '__main__':
    main()
