from torchvision.models import vgg16
import torch

class VGG16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vgg16()
    
    def forward(self, X):
        return self.model(X)

def wrap(model):
    class ModelClass(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
        
        def forward(self, X):
            return self.model(X)
    return ModelClass