import numpy as np
from torchvision import transforms

dataset_links = {
    'CIFAR-10': 'https://www.cs.toronto.edu/~kriz/cifar.html'
}

inv_trans = transforms.Compose([
    transforms.Normalize(mean=[ 0., 0., 0. ], std=[ 1/0.229, 1/0.224, 1/0.225 ]),
    transforms.Normalize(mean=[ -0.485, -0.456, -0.406 ], std=[ 1., 1., 1. ]),
])

def convert_tensor_for_display(tensor):
    return np.clip(np.transpose(inv_trans(tensor).numpy(), (1, 2, 0)), 0, 1)