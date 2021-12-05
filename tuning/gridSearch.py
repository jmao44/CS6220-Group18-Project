import torch
from torch import nn
from skorch import NeuralNetClassifier
import skorch
from sklearn.model_selection import GridSearchCV
import torch.nn
import torchvision.models as models
import numpy as np
import skorch.callbacks
import torch.optim

"""def update_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Update learning rate using LRBench at the beginning at each epoch. The LRBench Object is the the lrbench param of the skorch net
class LRBenchCallback(skorch.callbacks.Callback):
    def __init__(self, lrbench):
        super().__init__()
        self.lrbench = lrbench

    def on_epoch_begin(self, net, **kwargs):
        print("Updating Learning Rate with LRBench")
        epoch = len(net.history)
        optimizer = net.optimizer
        update_learning_rate(optimizer, self.lrbench.getLR(epoch))"""


def gridSearchHyperparameters(model, X, y, device='cpu', params=None, lrbenches=None, subset=None):
    # Wrap pytorch module as sklearn classifier to use GridSearchCV
    num_classes = int(y.max() + 1)
    skmodel = NeuralNetClassifier(
        model,
        max_epochs=10,
        lr=0.1,
        train_split=False,
        optimizer=torch.optim.Adam,
        verbose=False,
        criterion=torch.nn.CrossEntropyLoss(),
        callbacks=[],
        device=device
    )
    if subset is not None:
        X = X[:subset]
        y = y[:subset]
    print(X.shape)
    print(y.shape)
    # Default tuned hyperparameter values
    if params is None:
        if lrbenches is None:
            params = {
                'lr': [0.1, 0.01, 0.005, 0.001],
                'batch_size': [32, 64, 128, 256]
            }
        else:
            print('Using LRBench')
            params = {
                'callbacks': [[('lr_scheduler', skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda e: lrbench.getLR(e))), ] for lrbench in lrbenches],
                'batch_size': [32, 64, 128, 256]
            }
    gs = GridSearchCV(skmodel, params, scoring='accuracy', cv=3, refit=False)
    gs.fit(X, y)
    return gs.best_score_, gs.best_params_
