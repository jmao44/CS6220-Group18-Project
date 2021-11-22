import streamlit as st
import matplotlib.pyplot as plt
import torch
import train
import utils

from torch import nn
from torch import optim
from torchvision.utils import make_grid

# Page layout
######################################
## Page expands to full width
st.set_page_config(
    page_title='CS6220-Group18',
    layout='wide'
)

######################################
## Page Title
######################################
st.title('Deep Neural Network Performance Optimization using GridSearch and GTDLBench')
st.write('**CS 6220 Project Group 18:** Junyan Mao, Jintong Jiang, Yiqiong Xiao, Gabriel Wang, Andrew Wang')

######################################
## App Workflow Process
######################################
st.subheader('Process')
st.write('1. Upload dataset or use sample datasets')
st.write('2. Utilize GridSearch & GTDLBench to run ')

######################################
## Sidebar
######################################

# Step 1: select dataset
st.sidebar.header('Step 1: Pick your dataset')
# upload file
uploaded_file = st.sidebar.file_uploader('Upload your own data', type=['csv'])
st.sidebar.markdown('[Example CSV input file]()')

# select dataset
dataset_select = st.sidebar.selectbox(
    'Or, pick an example dataset from the list',
    ('CIFAR-10', 'Fashion-MNIST', 'MNIST', 'ImageNet')
)

# Step 2: select model
st.sidebar.header('Step 2: Pick your model')
model_select = st.sidebar.selectbox(
    'Pick a model',
    ('AlexNet', 'ResNet-18', 'VGG-16')
)

# Step 3: generate the optimal batch size
st.sidebar.header('Step 3: Generate the optimal batch size')
if st.sidebar.button('Generate'):

    if dataset_select == 'CIFAR-10' and model_select == 'AlexNet':
        model = train.init_alexnet()
        st.subheader('Model: {}'.format(model_select))
        st.code(model.eval())
        st.subheader('Dataset: [{}]({})'.format(dataset_select, utils.dataset_links[dataset_select]))

        with st.spinner('Loading dataset...'):
            train_loader, test_loader = train.load_data(dataset_name=dataset_select)
            data_iter = iter(train_loader)
            images, labels = data_iter.next()
            sample_image = utils.convert_tensor_for_display(make_grid(images))
            st.image(sample_image)

        st.subheader('Training:')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        epochs = 5
        patience = 3
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters())
        test_loss = []
        train_loss = []
        test_acc = []
        train_acc = []
        epoch_no_improve = 0
        prev_test_loss = None
        progress_bar = st.progress(0)
        for epoch in range(epochs):
            progress_bar.progress(epoch * 20)
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

            st.write(f"Epoch {epoch+1}/{epochs}.. "
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
        progress_bar.progress(100)
        model.eval()

        fig, ax = plt.subplots()
        ax.plot(range(1, epochs + 1), train_acc, 'g', label='Train Accuracy')
        ax.plot(range(1, epochs + 1), test_acc, 'b', label='Test Accuracy')
        ax.set_title('Train and Test Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()

        st.pyplot(fig)

        # plot_losses(epochs, train_loss, test_loss)
        # plot_acc(epochs, train_acc, test_acc)