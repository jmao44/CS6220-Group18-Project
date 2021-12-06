from torchvision.models import AlexNet
from torchvision.models import vgg16
import torch
from datasets import get_CIFAR_10
import gridSearch
from my_model import MyModel
import sys
import os
sys.path.append(os.curdir)
from LRBenchCustom.lr.piecewiseLR import piecewiseLR, LR

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel().to(device)
    X_train, y_train, X_test, y_test = get_CIFAR_10()
    lrbench1=LR({'lrPolicy': 'SINEXP', 'k0': 0.005, 'k1':0.01, 'l': 5, 'gamma':0.94})
    lrbench2=LR({'lrPolicy': 'POLY', 'k0': 0.005, 'k1':0.02, 'p':2, 'l':5})
    lrbenches=[lrbench1,lrbench2]
    print(gridSearch.gridSearchHyperparameters(model, X_train, y_train, device=device, lrbenches=lrbenches, subset=1280))