from torchvision.models import AlexNet
from torchvision.models import vgg16
import torch
from datasets import get_CIFAR_10
import gridSearch
from my_model import MyModel
from LRBench.lr.piecewiseLR import piecewiseLR


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
X_train, y_train, X_test, y_test = get_CIFAR_10()
lrbench1=piecewiseLR([11, ], [{'lrPolicy': 'SINEXP', 'k0': 0.1, 'k1':0.05, 'l': 5, 'gamma':0.94},
                                  {'lrPolicy': 'POLY', 'k0': 0.01, 'k1':0.05, 'p':1.2, 'l':5}])
lrbench2=piecewiseLR([2, ], [{'lrPolicy': 'SINEXP', 'k0': 0.1, 'k1':0.05, 'l': 5, 'gamma':0.94},
                                  {'lrPolicy': 'POLY', 'k0': 0.01, 'k1':0.05, 'p':1.2, 'l':5}])
lrbenches=[lrbench1,lrbench2]
print(gridSearch.gridSearchHyperparameters(model, X_train, y_train, device=device, lrbenches=None, subset=1280))