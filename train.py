
import torch
from torch import nn
from torch import optim
import pandas as pd
import numpy as np
from utils.utils import data_to_series, random_pandasDF
from modules.scheduler import CosineAnnealingWarmUpRestarts
from modules.models import LSTMNet, Conv1dLSTMNet
from modules.earlystop import EarlyStopping


if __name__ =="__main__":
    #data = pd.read_csv("data/time_series.csv", index_col='date', parse_dates=['date'])
    data = random_pandasDF(N=500, N_FEATURES=30)
    # Change dimension of data; ndim 2 --> 3
    TIME_STEP = 20
    X, y = data_to_series(data.drop(['y'], axis=1).values, data['y'].values, TIME_STEP)
    # Convert to tensor
    X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
    # Split dataset
    idx = int(len(data)*0.8)
    x_train, x_test, y_train, y_test = X[:idx], X[idx:], y[:idx], y[idx:]

    # Set up
    N_FEATURES = 30
    EPOCH = 5
    model = Conv1dLSTMNet(INPUT_DIM=N_FEATURES, TIME_STEP=TIME_STEP, HIDDEN_DIM=64, N_LAYER=1, OUTPUT_DIM=1)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=150, T_mult=1, eta_max=0.1, T_up=10, gamma=0.5)
    early_stopping = EarlyStopping(patience=5, verbose=True, path="Trained_TS_model.pt")

    # Training
    train_loss_avg = []
    val_loss_avg = []
    for i in range(EPOCH):

        train_loss = 0
        val_loss = 0

        model.train()
        for x, y in zip(x_train, y_train):
            optimizer.zero_grad()
            pred = model(x.view(1, x.size(0), x.size(1)))
            loss = loss_fn(pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(x_train)
        scheduler.step()
        train_loss_avg.append(train_loss)

        model.eval()
        for x, y in zip(x_test, y_test):
            pred = model(x.view(1, x.size(0), x.size(1)))
            loss = loss_fn(pred.squeeze(), y)
            val_loss += loss.item() / len(x_test)
        val_loss_avg.append(val_loss)

        print("Epoch {0}/{1}. Train loss {2:.4}. Val loss {3:.4}".format(i+1, EPOCH,train_loss, val_loss))

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

