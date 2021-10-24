
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from itertools import chain


class Evaluator:

    def __init__(self, model, PATH, device_type):
        self.PATH = PATH
        if device_type == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.load_state_dict()

    def __call__(self, test_dataset):
        self.testing(test_dataset)

    def load_state_dict(self):
        self.state_dict = torch.load(self.PATH, map_location=self.device)
        self.model.load_state_dict(self.state_dict)
        self.model.eval()

    def testing(self, test_dataloader):
        preds = []
        ys = []
        for x, y in test_dataloader:
            x = x.to(self.device)
            pred = self.model(x).data.max(1, keepdim=True)[1].squeeze()
            pred = pred.cpu().numpy()
            y = y.cpu().numpy()
            preds.append(pred)
            ys.append(y)
        self.pred = list(chain(*preds))
        self.labels = list(chain(*ys))
        acc = accuracy_score(self.pred, self.labels)
        matrix = confusion_matrix(self.pred, self.labels)
        print("Accuracy. {0:.4}".format(acc))
        print(matrix)

    def model_plot(self, test_dataloader, dim=2, markersize=1.5):

        for i, (x,y) in enumerate(test_dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)
            pred = pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            if i == 0:
                preds = pred
                Ys = y
            else:
                preds = np.append(preds, pred, axis=0)
                Ys = np.append(Ys, y)

        Y_set = set(Ys)
        colors = ['red', 'blue', 'green', 'black', 'yellow', 'purple']

        if dim == 2 or preds.shape[1] == 2:

            fig = plt.figure(figsize=(8,8))

            for i, label in enumerate(Y_set):
                idx = [i for i, x in enumerate(Ys) if x == label]
                selected = preds[idx]
                plt.scatter(selected[:,0], selected[:,1], c=colors[i], s=markersize, label=label)

            plt.grid()
            plt.legend()

        elif dim == 3:

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')

            for i, label in enumerate(Y_set):
                idx = [i for i, x in enumerate(Ys) if x == label]
                selected = preds[idx]
                ax.scatter(selected[:,0], selected[:,1], selected[:,3], c=colors[i], s=markersize, label=label)

            plt.grid()
            plt.legend()





