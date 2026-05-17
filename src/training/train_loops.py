
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

def train(model: nn.Module, 
          dataloader: DataLoader, 
          horizon: int, 
          optimizer: torch.optim.Optimizer, 
          criterion = nn.MSELoss(), 
          nepochs: int = 50):
    """Train a neural network model

    Args:
        model: Model to train.
        dataloader: dataloader containing features and targets
        horizon: number of time steps into the future to forecast
        optimizer: optimization algorithm
        criterion: loss function
        nepochs: Number of training epochs

    Returns:
        A tuple of (losses, accuracies). Each is a list of values recorded
        at each epoch during training.
    """


    losses = []
    for epoch in tqdm(range(nepochs)):
        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            optimizer.zero_grad()

            predictions = model(X_batch, horizon=horizon)

            loss = criterion(predictions, y_batch)

            loss.backward()

            optimizer.step()

            losses.append(loss.item())

        if (epoch % 10) == 0:
            print(f'epoch: {epoch}, loss: {loss.item():.3f}')

    return losses

def train_with_early_stopping(model: nn.Module, 
                              train_dataloader: DataLoader,
                              val_dataloader: DataLoader,
                              horizon: int, 
                              optimizer: torch.optim.Optimizer, 
                              criterion = nn.MSELoss(), 
                              max_epochs: int = 100,
                              patience: int = 30):
    """Train a neural network model with early stopping

    Args:
        model: Model to train.
        dataloader: dataloader containing features and targets
        horizon: number of time steps into the future to forecast
        optimizer: optimization algorithm
        criterion: loss function
        max_epochs: Maximal number of training epochs
        patience: Number of training epochs to wait until you stop training due to no improvement on validation loss

    Returns:
        A tuple of (losses, accuracies). Each is a list of values recorded
        at each epoch during training.
    """
    losses_val = []
    for (X_val_batch, y_val_batch) in val_dataloader:
        y_pred_val = model(X_val_batch, horizon=horizon)
        losses_val.append(criterion(y_pred_val, y_val_batch).item())
    best_loss_val = np.mean(losses_val)
    wait = 0
    stopped_epoch = max_epochs

    losses_train = []
    losses_val = []
    for epoch in tqdm(range(max_epochs)):
        batch_losses = []
        for (X_batch, y_batch) in train_dataloader:
            optimizer.zero_grad()

            predictions = model(X_batch, horizon=horizon)

            loss = criterion(predictions, y_batch)

            loss.backward()

            optimizer.step()

            batch_losses.append(loss.item())

        losses_train.append(np.mean(batch_losses))

        if (epoch % 10) == 0:
            print(f'epoch: {epoch}, loss: {loss.item():.3f}')


        losses_val_batches = []
        for (X_val_batch, y_val_batch) in val_dataloader:
            y_pred_val = model(X_val_batch, horizon=horizon)
            losses_val_batches.append(criterion(y_pred_val, y_val_batch).item())
        loss_val = np.mean(losses_val_batches)
        losses_val.append(loss_val)

        if loss_val < best_loss_val:
            best_loss_val = loss_val
            wait = 0
        else:
            wait += 1

        if wait > patience:
            stopped_epoch = epoch
            break

    return losses_train, losses_val, stopped_epoch