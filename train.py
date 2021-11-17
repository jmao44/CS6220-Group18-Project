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

# Globally set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(subset=True, dataset_name='CIFAR10'):
    # Properly transform the images to work with AlexNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
    resize = transforms.Resize((224, 224))
    my_transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])

    # TBA other datasets
    if dataset_name == 'CIFAR10':
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
    # Hardcoded subset used for example purposes, only use 500 images for train/test
    if subset:
        subset_list = list(range(0, 500))
        training_data = torch.utils.data.Subset(training_data, subset_list)
        test_data = torch.utils.data.Subset(test_data, subset_list)

    # TBA use GridSearch to modify batch sizes
    train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    return (train_loader, test_loader)


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

def train(epochs=5, patience=3):
    train_loader, test_loader = load_data()
    model = init_alexnet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters())

    test_loss = []
    train_loss = []
    test_acc = []
    train_acc = []
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

    model.eval()
    plot_losses(epochs, train_loss, test_loss)
    plot_acc(epochs, train_acc, test_acc)
    return model

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






