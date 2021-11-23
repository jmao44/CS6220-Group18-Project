import torch
import torchvision
import torchvision.transforms as transforms

def get_CIFAR_10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    X_train, y_train = zip(*[data for data in trainset])
    X_train, y_train = torch.stack(X_train), torch.Tensor(y_train).type(torch.LongTensor)
    print(X_train.shape)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    X_test, y_test = zip(*[data for data in trainset])
    X_test, y_test = torch.stack(X_test), torch.Tensor(y_test).type(torch.LongTensor)

    return X_train, y_train, X_test, y_test
