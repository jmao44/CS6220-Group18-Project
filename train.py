import torch
from torch.utils.data import Dataset
import torchvision.models as models
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import tuning.gridSearch
from tuning.LRBenchCustom.lr.piecewiseLR import piecewiseLR, LR

# Globally set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(subset=True, dataset_name='CIFAR-10'):
    # Properly transform the images to work with AlexNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
    resize = transforms.Resize((224, 224))
    my_transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])

    if dataset_name == 'CIFAR-10':
        training_data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=my_transform
        )

        test_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=my_transform
        )
    elif dataset_name == 'Fashion-MNIST':
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=my_transform
        )

        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=my_transform
        )
    elif dataset_name == 'MNIST':
        training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=my_transform
        )

        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=my_transform
        )
    # TBA User uploaded dataset
    else:
        return None

    # Hardcoded subset used for example purposes, only use 500 images for train/test
    if subset:
        subset_list = list(range(0, 500))
        training_data = torch.utils.data.Subset(training_data, subset_list)
        test_data = torch.utils.data.Subset(test_data, subset_list)

    # TBA use GridSearch to modify batch sizes
    train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    X_train, y_train = zip(*[data for data in training_data])
    X_train, y_train = torch.stack(X_train), torch.Tensor(y_train).type(torch.LongTensor)
    X_test, y_test = zip(*[data for data in test_data])
    X_test, y_test = torch.stack(X_test), torch.Tensor(y_test).type(torch.LongTensor)
    return X_train, y_train, X_test, y_test, train_loader, test_loader


def init_alexnet(num_classes=10):
    model = models.alexnet(pretrained=True)
    model.eval()

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Set last layers as linear classifier based on num_classes
    model.classifier[6] = nn.Linear(4096, num_classes)
    model.to(device)

    print(model)

    return model

def init_resnet(num_classes=10):
    model = models.resnet18(pretrained=True)
    model.eval()

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Set last layers as linear classifier based on num_classes
    model.fc = nn.Linear(512, num_classes)
    model.to(device)

    print(model)

    return model

def init_vgg(num_classes=10):
    model = models.vgg16(pretrained=True)
    model.eval()

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Set last layers as linear classifier based on num_classes
    model.classifier[6] = nn.Linear(4096,num_classes)
    model.to(device)

    print(model)

    return model

def get_model(model_name):
    if model_name == 'AlexNet':
        model = init_alexnet()
    elif model_name == 'ResNet-18':
        model = train.init_resnet()
    elif model_name == 'VGG-16':
        model = train.init_vgg()
    return model

def gridsearch(model_select):
    X_train, y_train, _, _, _, _ = load_data()
    model = get_model(model_select).to(device)

    lrbench1=LR({'lrPolicy': 'SINEXP', 'k0': 0.005, 'k1':0.01, 'l': 5, 'gamma':0.94})
    lrbench2=LR({'lrPolicy': 'POLY', 'k0': 0.005, 'k1':0.02, 'p':2, 'l':5})
    lrbenches=[lrbench1,lrbench2]
    best_score, best_params, gs = tuning.gridSearch.gridSearchHyperparameters(model, X_train, y_train, device=device, lrbenches=lrbenches)
    batch_size = best_params['batch_size']
    if 'lr' in best_params.keys():
        lr = best_params['lr']
    elif 'callbacks' in best_params.keys():
        print(best_params['callbacks'])
        lr = best_params['callbacks'][0][1].lrbench.lrParam['lrPolicy']
    return best_score, batch_size, lr

def train(given_model, epochs=5, patience=3):
    _, _, _, _, train_loader, test_loader = load_data()
    model = given_model
    criterion = nn.CrossEntropyLoss()

    if hasattr(model, 'classifier'):
        optimizer = optim.Adam(model.classifier.parameters())
    elif hasattr(model, 'fc'):
        optimizer = optim.Adam(model.fc.parameters())
    test_loss = []
    train_loss = []
    test_acc = []
    train_acc = []
    res =[]
    epoch_no_improve = 0
    prev_test_loss = None

    for epoch in range(epochs):
        for data, targets in iter(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            results = model.forward(data)
            # Computes loss
            loss = criterion(results, targets)

            # Updates model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate after every epoch
        test_loss_val = 0
        test_accuracy = 0
        train_loss_val = 0
        train_accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                scores = model.forward(inputs)
                batch_loss = criterion(scores, labels)
                train_loss_val += batch_loss.item()

                # Calculate accuracy
                top_p, top_class = scores.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                scores = model.forward(inputs)
                batch_loss = criterion(scores, labels)
                test_loss_val += batch_loss.item()

                # Calculate accuracy
                top_p, top_class = scores.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        final_test_loss = test_loss_val/len(test_loader)
        final_train_loss = train_loss_val/len(train_loader)
        final_test_acc = test_accuracy/len(test_loader)
        final_train_acc = train_accuracy/len(train_loader)

        test_loss.append(final_test_loss)
        train_loss.append(final_train_loss)
        test_acc.append(final_test_acc)
        train_acc.append(final_train_acc)

        print(f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {final_train_loss:.3f}.. "
                f"Test loss: {final_test_loss:.3f}.. "
                f"Train accuracy: {final_train_acc:.3f}.. "
                f"Test accuracy: {final_test_acc:.3f}")

        if prev_test_loss:
            # Check early stopping
            if prev_test_loss < final_test_loss:
                epoch_no_improve += 1
            else:
                epoch_no_improve = 0

        prev_test_loss = test_loss[-1]

        if epoch_no_improve >= patience:
            print("EARLY STOPPING")
            break

        model.train()
    res.append([int(i)+1 for i in range(epochs)])

    res.append(train_acc)
    res.append(train_loss)

    res.append(test_acc)
    res.append(test_loss)
    model.eval()
    plot_losses(epochs, train_loss, test_loss)
    plot_acc(epochs, train_acc, test_acc)
    return epochs, train_loss, test_loss,train_acc, test_acc, res

def plot_losses(epochs, train_loss, test_loss):
    epochs_range = range(1,epochs+1)
    plt.plot(epochs_range, train_loss, 'g', label='Training Loss')
    plt.plot(epochs_range, test_loss, 'b', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./visualizations/loss.png')

def plot_acc(epochs, train_acc, test_acc):
    epochs_range = range(1,epochs+1)
    plt.plot(epochs_range, train_acc, 'g', label='Training accuracy')
    plt.plot(epochs_range, test_acc, 'b', label='Test accuracy')
    plt.title('Training and Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./visualizations/accuracy.png')






