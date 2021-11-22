import streamlit as st
import pandas as pd
import numpy as np
import train
import utils

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