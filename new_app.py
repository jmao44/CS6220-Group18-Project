import streamlit as st
import pandas as pd
import numpy as np

# Page layout
######################################
## Page expands to full width
st.set_page_config(page_title='Determining Feature Importance - Machine Learning App',
                   layout='wide')

######################################
## Page Title
######################################
st.title('Deep Neural Network Performance Optimization using GridSearch and GTDLBench')
st.write('**CS 6220 Project Group 18:** Junyan Mao, Jintong Jiang, Yiqiong Xiao, Gabriel Wang, Andrew Wang')

######################################
## App Workflow Process
######################################
st.header('Process')
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
