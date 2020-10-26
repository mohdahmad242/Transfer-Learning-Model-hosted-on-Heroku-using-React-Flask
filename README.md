# Transfer Learning Model hosted on Heroku using React & Flask

## Introduction
Online platforms have been acting as an open platform for people to express their views. But sometimes, these expressions lead to discussions, and these discussions might turn into debates when a difference in opinion arises. It takes no time for such discussion forums to get dirty and people to start spreading negativity. There exist many measures to counter these negative comments, Artificial Intelligence Classification and Detection being one of them. Is it not wonderful that a computer that only understands 0 and 1, the famous binary numbers can also be trained to have a human level perception of texts?
Today, we are going to teach you a way to create your own text classifier which can be used to check whether the sentiment a sentence carries is positive or negative. The implementation of this deep learning model is based on RoBERTa, which is a Robustly Optimized BERT Pretraining approach. This pre-trained RoBERTa model is first pre-trained on the IMDb dataset, after which the learning was transferred to the SST2 dataset. To make the model presentable to a user, we will be deploying it on a front-end application created using ReactJS. The contents of this tutorial are given as follows:
1. Pre-training
  - Download the pre-trained RoBERTa model
  - Set up the IMDb dataset into the input format of RoBERTa
  - Use the IMDb dataset to pre-train the model
  - Save the model’s weight to use for transfer learning
2. Use the saved weights to implement Transfer Learning
  - Use the saved weights from Step 1 and recreate the RoBERTa model
  - Set up the IMDb into the input format of RoBERTa
  - Use the recreated model and train it using the IMDb dataset changing the final layer only
  - Test the model using the reserved test set and save the results
3 .Deploy the model on Heroku server
  - Create a Flask backend
  - Create a React front-end
  - Synchronize the front-end and back-end
  - Deploy the project using Heroku
