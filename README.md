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
3. Deploy the model on Heroku server  
    - Create a Flask backend
    - Create a React front-end
    - Synchronize the front-end and back-end
    - Deploy the project using Heroku  
The whole tutorial is broken down into these 3 sections to make it easier and simpler to understand. But before we begin, here is the explanation of a few terminologies which will be frequently used throughout the tutorial.  

## Important Terminologies
- **MAX_SEQ_LEN**: This hyperparameter is used to define the maximum length of a sentence which will be taken into consideration. That is, the maximum number of words from each review which will be provided to the model.
- **PRE_TRAINING_TRAIN_BATCH_SIZE**: The number of reviews after which the model updates its weights. In simple terms, it is the number of reviews after which loss is calculated and weights are revised on the basis of loss. A smaller training batch size means that the weights are updated more frequently.
- **PRE_TRAINING_VAL_BATCH_SIZE**: The number of reviews which will be given to the model during the validation of the model. During the validation period, the weights of the model are frozen to check its performance against a completely new dataset.
- **PAD_INDEX**: As discussed in the MAX_SEQ_LEN, all reviews must have the same length when they are fed into the model. While the longer reviews are trimmed, the shorter ones are padded by adding 0 either at the beginning or at the end of the sentence. This task is done by PAD_INDEX.
- **UNK_INDEX**: It might be possible that not all words present in the reviews might be available in the vocabulary files, this can be due to multi-lingual reviews, authors making typos while posting the reviews or many such reasons. The UNK_INDEX is used to handle these unknown or out-of-vocabulary words by changing all them to zero.
- **Field**: Field is a pre-defined class model in Pytorch which is used to create a datatype that converts sentences or words to tensors. It contains the instructions like how to tokenize the sentences, maximum sentence length to be taken into consideration, type of padding, how to handle unknown vocabulary, etc. More information about field can be found at the official documentation of Pytorch [here](https://pytorch.org/text/data.html#field).
- **TabularDataset**: TabularDataset creates a dataset in either CSV, TSV or JSON format given the path of the file and the Fields to be considered. We have used the CSV format with a 0.8, 0.1, 0.1 split for train, test and validation dataset for the IMDb dataset, and for the SST2 dataset the ratios were set to 0.7, 0.1 and 0.2. To know more about TabularDataset, try the Pytorch official documentation [here](https://pytorch.org/text/data.html#tabulardataset).
- **Iterator**: As discussed in the PRE_TRAINING_TRAIN_BATCH_SIZE, the training dataset is divided into batches before being fed to the network. This division of batches of the training, validation or testing data is performed by the Iterator function which loads the batches from the provided dataset.
This concludes our discussion for the important terminologies which will be used repeatedly. Now we move forward to the dataset being used.

## Dataset
#### IMDb Dataset
The IMDb dataset is made available at [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/), it has 25000 highly polar movie reviews for training and other 25000 for testing. The high polar nature of these reviews makes it a good pre-training dataset as the features extracted from these texts would relate highly to the positive or negative nature of the reviews. In simple words, the more polar or contrasting the two classes are, the better will be the pre-training model in distinguishing between the classes. Also, the class distribution of these reviews is almost equal, i.e. the number of positive reviews is approximately equal to the number of negative reviews, hence no class biasness would occur during the pre-training of the model. We took the complete 50000 tweets of the dataset and divided them in the ratio of 8:1:1 for training, validation and testing.  

## Transfer Learning
When dealing with many classification or segmentation problems in deep learning, the available dataset might not be at par with the problem. One of the major problems is the availability of very limited datasets. There exist many solutions to deal with these problems, Transfer learning is one of them. So, the very first question is what is transfer learning? Transfer learning is a research problem in machine learning that focuses on storing the knowledge a model gains while solving a problem and then use that knowledge to solve a similar problem. For example, a model which already knows how to differentiate between the sentiment of reviews can be used to differentiate between sentiments overall with just a little buff. This buff is known are re-training the model. In other words, we first train the model on a large dataset available for a similar task as our problem, save that knowledge, use that knowledge and train the model on a part of our task’s dataset and then test its performance. Transfer learning has shown great advancements in the field of cancer subtype discovery, building utilization, general game playing, text classification and other tasks. 
In our tutorial, we take a Pre-trained RoBERTa model, transfer that learning to the IMDb dataset, and then further transfer that learning to our IMDb dataset. Now that Transfer learning is dealt with, let us begin with the pre-training of our model. Here are some papers which discuss Transfer Learning in various Deep Learning tasks:
- [ADVERSARIALLY ROBUST TRANSFER LEARNING](https://openreview.net/pdf?id=ryebG04YvB)
- [A Survey on Deep Transfer Learning](https://arxiv.org/abs/1808.01974)
- [Self-taught Learning: Transfer Learning from Unlabeled Data](https://dl.acm.org/doi/abs/10.1145/1273496.1273592)  

## Model Architecture
The model we have used is a RoBERTa model with some added layers. The layers added to the RoBERTa model are as follows:
- Dropout Layer having 0.3 dropout rate
- Linear Layer having shape (768, 256)
- Layer Normalization
- Linear Layer having shape (256, 64)
- Layer Normalization
- Dropout Layer having 0.3 dropout rate
- Linear Layer having shape (64, 2)  
The function of all these layers is explained as follows:
- Dropout Layer: A dropout layer removes a fraction of the inputs at random. The fraction of inputs which are to be removed is given by the dropout rate. So, a 0.3 dropout rate means that 30% of the inputs to this layer will not be given forward, and the order of these excluded neurons is chosen at a random so that no bias occurs.
- Linear Layer: A linear layer has two input parameters, the number of input units and the number of output units. It takes the number of input units and fits them into the multiple linear equation to give the output units. We have three additional linear layers attached to out RoBERTa model which have shapes (768, 256), (256, 64) and (64, 2) the last linear layer gives two outputs, which are taken as the probability of the input belonging to the two respective classes.
- Layer Normalization: A normalization layer uses the mean and the variance of the inputs and normalizes the distributions of the layer, i.e. it makes sure that all the units of the layer are normalized to facilitate smoother gradients and faster training. We have used the normalization layer after the first two linear layers in our model.
The architecture of the customized RoBERTa model that we have used is shown in the figure below.
We now proceed to the pre-training of our customized model.

## Pre-training Customized RoBERTa
The pre-training steps are quite simple to understand. The pseudo-code for the training can be given as:  
```
epoch starts
training and validation loss set to zero
for batch in batches:
	model is fit to batch
	losses are calculated using loss function
	derivatives are calculated
	derivatives are subtracted from weight matrices
	batch ends
weights are frozen
validation set is used to calculate validation performance and loss
if current validation loss < previous least validation loss:
	save weights
weights are unfreezed and epoch ends
```
This pseudo code is very simple to understand. Now let’s look at the code and the explanation. As you can see, we have set  
```
param.requires_grad = False
```
in the pre-training step. This is to ensure that during the pre-training the gradients of the RoBERTa model are not changed or updated, only the layers which follow the original pre-trained model are trained as per the IMDb dataset. The criterion/ loss function is set to Cross Entropy Loss and we have set the best validation loss to infinity. The use of this best validation loss will be explained further. The train loop begins and we set the train loss and valid loss to zero each. We then use the train iterator to load the first training batch and calculate the predictions using
```
y_pred = model(input_ids=source, attention_mask=mask)
```
Then the losses are calculated which take the predicted values and the original values related to each example in the training batch. The ```loss.backward()``` function initiates the backpropagation and the weights are updated. We have also used a ```scheduler.step()``` to keep track of our learning rate. We have used a decaying learning rate to decrease the learning rate as the training progresses, this helps in overcoming overfitting. We then store the training losses, and increase the number of global steps by 1. At the end we check if the global step is equal to our validation period, to test the model on our validation data. Before we proceed with our validation steps, we freeze the gradients of our model using
```
with torch.no_grad():
```
As we don’t want our model to update the gradients during the validation period. We then calculate the predictions and calculate the validation loss. We then calculate this validation loss to our best validation loss, in case our validation loss comes out to be less, the model is saved and the training continues as usual. If the loss is more, the model is not saved and the training continues as usual. This makes sure that we only save those checkpoints, where the model best fits our validation dataset, helping us to curb overfitting. Once the training is complete, we set the training parameters of RoBERTa back to true. This was the explanation of the pretrain function. The next piece of code, sets the hyperparameters and calls the function to initiate the pre-training. The model is run for 12 epochs, and the model having best validation accuracy is saved. In the next section we will use this saved model to create our classifier and test its performance.

## Creating Classifier with Transfer Learning
We first start with setting up the hyperparameters just like we did for the pre-training part. These hyperparameters and their values remain the same. Since the dataset remians the same in this step, the pre-processing step for the dataset remains the same as above.   
We then create the exact same replica of our customized RoBERTa model. In the next piece of code, we create lists to store the training loss, validation loss and global steps. These shall be used to plot the trends in losses during the training period. Then we initiate the classifier, only to use the ```torch.load()``` to load the saved model instead of a random initiation. This is known as creating the transfer model. Since the architectures of both the models are same, we don’t need to make any changes to the load function. Once again, we set the best validation loss to infinity and the loss function to Cross Entropy Loss. The epochs are started similarly, and the training progresses. Also, it must be noticed that, this time we have not frozen the RoBERTa layers, and allowed them to train as well. This ensures that the model fits itself to the dataset well. The rest of the training procedure is exactly same as that of the pre-training method. We freeze the weights during the testing of validation set, and store both the training and testing losses in our lists. The model having least validation loss is saved and then can be deployed on the Web Application, which we will discuss in the further sections of our tutorial.  
In the last piece of our code, we use the final model for the test set. We use the ```torch.no_grad()``` to freeze the weights again, and test the model. The predicted labels are then used to generate a classification report using the sklearn library. The model is saved using ```torch.save(model.state_dict(), "modelName.pth")```

## Deploy Deep Learning Model using Flask
Now that our classification model is ready and saved, we can deploy it using a simple Flask application. Before we start with the code, let us see what Flask is. Flask is a simple web application framework for Python, which allows the user to write applications without worrying about protocol or thread management. You can learn more about Flask from their official documentation [here](https://flask.palletsprojects.com/en/1.1.x/).  
In our Flask application, we add the saved model, and a [predict.py](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Webapp/Flask/ml_model/predict.py) file that makes the predictions. This pred.py file contains the architecture of the model, that would load the saved weights of the model. Since, we had space limitations for deployment, we used a simple LSTM model to demonstrate this part. The code remains exactly same for deploying RoBERTa, only the class ```LSTM()``` would be replaced by our RoBERTa model. This model that we have used is explained in the end for your understanding. In the pred.py file, we create the model and then load the saved weights using ```model.load_state_dict()``` function, then we have a function ```pred()``` that uses the model to predict the incoming review request. The front-end sends the review using a form which has been created using Prediction component which will be explained in the next section. For now let us just think of it as simple input parameter sent by the front-end application. The model then predicts the sentiment and returns the sentiment. The app.py file contains the encapsulation of our request and post API. We use the POST API to send the final sentiment of the tweet to our front-end. Now that our back-end is complete, we can proceed with our Front-end application. The description of our front-end application is mentioned in the next section.  

## Creating a React Front-End
We have created a very simple React front-end for our tutorial. To initiate the process, we first create an empty folder and begin with the command
```
npx create-react-app applicationName
```
This would download the required libraries to the folder and you can initiate the basic application using
```
cd my-app
npm start
```
The next step would be to create the required components for our application. Before we get into the details of components, let us see what components in React actually mean. As per the official React documentation, components are similar to Javascript functions, which take arbitrary inputs and return React elements which describe what should appear on the screen. In simple language they are pieces of code, that can be used repeatedly and exist independently. More details about components can be found [here](https://reactjs.org/docs/react-component.html). We create the components in the in the [componenets folder](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/tree/main/Webapp/React/src/components). We have created two components named as Example and Prediction. The [example.js](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Webapp/React/src/components/example.js) component contains the details of the examples visible on our webpage, while the [prediction.js](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Webapp/React/src/components/prediction.js) component contains a form, which allows us to submit a sentence for testing the sentiment. This testing of the sentence is executed using our custom created API, which sends a request to the server containing the sentence. The server receives the request, tests the sentence using the classifier and then returns the sentiment using get request. The visuals of the project can be changed by making changes in the from the [index.css](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Webapp/React/src/index.css) file.  

## LSTM Model used in Web Application
Since, Heroku only allows a total space of 500Mb to be uploaded, we could not host our final RoBERTa model there due to its humongous size. To demonstrate the working of our web application we decided to proceed with a lighter LSTM model. You can find the architecture of the model in [predict.py](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Webapp/Flask/ml_model/predict.py) file. The model is defined in the class ```LSTM()```. The model has an embedding layer of size 500, which takes the sentences and returns the vocabulary embedding of the words. Then we have a bidirectional LSTM layer having 128 units, the bidirectional nature of the layer allows the model to look at the sentence in both front and back order. We then have a Dropout layer which drops 30% of the words LSTM output units during training. This makes our model more robust and would increase the accuracy during testing as explained above. The final layer is a Linear layer having single output. If the output is smaller than 0.5, we annotate it to the negative class, while having a value greater than 0.5 means the sentence belongs to the positive class. This brings us to  the end of our tutorial.

## Summarizing it all
The project is finally complete. To summarize it, here is the encapsulated version of it all. We first created a pre-trained model of our choice, which was transferred to an exact replica, the replica was trained on a similar dataset (same dataset) in our tutorial. This model was then saved for deployment. A flask application was created which had functions to handle the post and get requests from the front-end. The flask application has a prediction.py file which loads the saved model and uses it to make predictions. Then we created a front-end using React, where the user can enter reviews through a form and get the desired output sentiment. The demonstration for the latter part was done using a simple LSTM model due to limitations in space. The final application was then hosted on Heroku server.

