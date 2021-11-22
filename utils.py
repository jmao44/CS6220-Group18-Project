import numpy as np
import streamlit as st
from torchvision import transforms

DATASET_LINKS = {
    'CIFAR-10': 'https://www.cs.toronto.edu/~kriz/cifar.html'
}

SAMPLE_DATASETS = (
    'CIFAR-10', 'Fashion-MNIST', 'MNIST', 'ImageNet'
)

SAMPLE_MODELS = (
    'AlexNet', 'ResNet-18', 'VGG-16'
)

def convert_tensor_for_display(tensor):
    inv_trans = transforms.Compose([
        transforms.Normalize(mean=[ 0., 0., 0. ], std=[ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean=[ -0.485, -0.456, -0.406 ], std=[ 1., 1., 1. ]),
    ])
    return np.clip(np.transpose(inv_trans(tensor).numpy(), (1, 2, 0)), 0, 1)


# return a bunch of empty placeholders
def initialize_placeholders(num):
    l = [st.empty() for _ in range(num)]
    return l

def empty_placeholders(placeholders):
    for p in placeholders:
        p.empty()