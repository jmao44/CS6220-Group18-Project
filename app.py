import streamlit as st
import matplotlib.pyplot as plt
import torch
import train
import utils

from torch import nn
from torch import optim
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
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
st.title('Interactive Deep Neural Network Performance Optimization using GridSearch and LRBench')
st.write('**CS 6220 Project Group 18:** Junyan Mao, Jintong Jiang, Yiqiong Xiao, Gabriel Wang, Andrew Wang')

######################################
## App Workflow Process
######################################
st.subheader('Introduction')
st.write('We created a AutoML platform that optimizes deep neural network performance. It leverages GridSearchCV and LRBench to optimize the hyperparameters of ML models. This platform provides an interactive way for users to select their choice of dataset and model and it visualizes the optimization, training, and performance analysis process.')
st.write("Follow the sidebar, to try it out!")

######################################
## Sidebar
######################################

# Step 1: select dataset
st.sidebar.header('Step 1: Pick your dataset')
# upload file
#uploaded_file = st.sidebar.file_uploader('Upload your own data', type=['csv'])
#st.sidebar.markdown('[Example CSV input file]()')

# select dataset
dataset_select = st.sidebar.selectbox('Pick an example dataset from the list', utils.SAMPLE_DATASETS)

# Step 2: select model
st.sidebar.header('Step 2: Pick your model')
model_select = st.sidebar.selectbox('Pick a model', utils.SAMPLE_MODELS)
batch_size_select = st.sidebar.number_input(
    label='Enter a batch size',
    value=64,
    step=1
)

# placeholders
placeholder_list = utils.initialize_placeholders(10)
model_name_ph, model_struct_ph, \
dataset_ph, dataset_sample_ph, \
training_ph, train_progress_bar_ph, \
info_table_ph, \
epoch_ph, acc_plot_ph, loss_plot_ph = placeholder_list

# Step 3: generate the optimal batch size
st.sidebar.header('Step 3: Select Action')
action_select = st.sidebar.selectbox('Train a model or search hyperparameters', utils.SAMPLE_ACTIONS)
if st.sidebar.button('Start'):
    utils.empty_placeholders(placeholder_list) # reset placeholders
    # Select Model
    if model_select == 'AlexNet':
        model = train.init_alexnet()
        model_name_ph.subheader('Model: {}'.format(model_select))
        model_struct_ph.code(model.eval())
    elif model_select == 'ResNet-18':
        model = train.init_resnet()
        model_name_ph.subheader('Model: {}'.format(model_select))
        model_struct_ph.code(model.eval())
    elif model_select == 'VGG-16':
        model = train.init_vgg()
        model_name_ph.subheader('Model: {}'.format(model_select))
        model_struct_ph.code(model.eval())

    # Select Dataset
    if dataset_select == 'CIFAR-10':
        dataset_ph.subheader('Dataset: [{}]({})'.format(dataset_select, utils.DATASET_LINKS[dataset_select]))
        with st.spinner('Loading dataset...'):
            _, _, _, _, train_loader, test_loader = train.load_data(batch_size_select, dataset_name=dataset_select)
            data_iter = iter(train_loader)
            images, labels = data_iter.next()
            sample_image = utils.convert_tensor_for_display(make_grid(images))
            dataset_sample_ph.image(sample_image)
    elif dataset_select == 'Fashion-MNIST':
        dataset_ph.subheader('Dataset: [{}]({})'.format(dataset_select, utils.DATASET_LINKS[dataset_select]))
        with st.spinner('Loading dataset...'):
            _, _, _, _, train_loader, test_loader = train.load_data(batch_size_select, dataset_name=dataset_select)
            data_iter = iter(train_loader)
            images, labels = data_iter.next()
            sample_image = utils.convert_tensor_for_display(make_grid(images))
            dataset_sample_ph.image(sample_image)
    elif dataset_select == 'MNIST':
        dataset_ph.subheader('Dataset: [{}]({})'.format(dataset_select, utils.DATASET_LINKS[dataset_select]))
        with st.spinner('Loading dataset...'):
            _, _, _, _, train_loader, test_loader = train.load_data(batch_size_select, dataset_name=dataset_select)
            data_iter = iter(train_loader)
            images, labels = data_iter.next()
            sample_image = utils.convert_tensor_for_display(make_grid(images))
            dataset_sample_ph.image(sample_image)

    if action_select == 'Optimize':
        training_ph.subheader('Training Results:')
        with st.spinner('Running GridSearch and finding optimal parameters...This may take a long time...'):
            best_score, batch_size, lr = train.gridsearch(model_select, batch_size_select)
            best_score = float(best_score) * 100
            data = {'Best Score': [best_score], 'Best Batch Size': [batch_size], 'Best Learning Rate Policy': [lr]}
            df = pd.DataFrame(data)
            info_table_ph.table(df)
    elif action_select == 'Train':
        training_ph.subheader('Training Results:')
        with st.spinner('Running and gathering data...This may take a few minutes...'):
            epoch, train_loss, test_loss, train_acc, test_acc, res = train.train(model, batch_size=batch_size_select)
        fig, ax = plt.subplots()
        ax.plot(range(1, epoch + 1), train_acc, 'g', label='Train Accuracy')
        ax.plot(range(1, epoch + 1), test_acc, 'b', label='Test Accuracy')
        ax.set_title('Train and Test Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        acc_plot_ph.pyplot(fig)

        fig, ax = plt.subplots()
        ax.plot(range(1, epoch + 1), train_loss, 'g', label='Train Loss')
        ax.plot(range(1, epoch + 1), test_loss, 'b', label='Test Loss')
        ax.set_title('Train and Test Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        loss_plot_ph.pyplot(fig)

        col_data = ['Num of epochs', 'Train Accuracy', 'Train Loss', 'Test Acurracy', 'Test Loss']
        data = np.array(res)
        df = pd.DataFrame(data.transpose(), columns=col_data)
        info_table_ph.table(df)
