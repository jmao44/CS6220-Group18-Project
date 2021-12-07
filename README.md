# Interactive Deep Neural Network Performance Optimization using GridSearch and LRBench 
**CS 6220 Group 18**: Junyan Mao, Jintong Jiang, Yiqiong Xiao, Gabriel Wang, Andrew Wang

## Introduction
Our project is an AutoML framework + Web App that optimizes deep neural network performance. Our AutoML framework trains popular models and optimizes their hyperparameters with [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and learning rate tuning with [LRBench](https://github.com/git-disl/LRBench) to produce the most optimal model. Our web app interacce provides an interactive experience for users to understand the optimization process with dataset upload functionality, model selection, and visualizations.

## Installation
If you would like to run our app locally, make sure you have Python3 and virtualenv installed:

First, set up a virtual environment

`python3 -m venv env`


`source env/bin/activate`

Download dependencies

`pip install -r requirements.txt`

Run streamlit app

`streamlit run app.py`

To speed up the training and optimization process, you can run this app within [Google Colab](https://colab.research.google.com)

## Live Deployment

https://share.streamlit.io/jmao44/cs6220-group18-project/app.py
