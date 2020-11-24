# Transfer Learning Model hosted on Heroku using React & Flask
> Transfer Learning model using RoBERTa on IMDb dataset deployed on React and Flask.  
> Try out [here](https://imdbmovienew.herokuapp.com/), Run on Google Collab [here](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Notebook/roberta.ipynb)

## Overview
 
<p align="center">
  <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/finalResult.gif">
</p>  

In this tutorial we will create a text classifier using RoBERTa model in PyTorch and deploy it using Heroku on a web application created using Flask and React JS. The topics covered in our tutorial are:
* Creating a basic Text Classifier
* Creating a transfer learning model using RoBERTa
* Deploying the model on a web application 

## Pre-Requisities
To implement the complete project you will need the following:
* Create a [GitHub account](https://github.com/join)
* Create a [Heroku account](https://signup.heroku.com/)
* Have knowledge of PyTorch and Deep Learning, follow the starter project [here]()

## Introduction
In our tutorial we discuss how to implement [Transfer Learning](https://ruder.io/transfer-learning/index.html#applicationsoftransferlearning) using PyTorch library and then deploy the model on a web application. Before we begin let us have a look at what transfer learning is. In case you know about it, please proceed to the [next section](). In our tutorial we have assumed that you have some previous knowledge of deep learning, in case you are new, go through our [starter project]() first.

### Transfer Learning
* Focuses on using stored knowledge from previous training and using it on a similar but different dataset.
* Extracts features from a relatively large dataset, this step is called pre-training and then uses this knowledge on the target dataset.
* Pre-training usually done on a large dataset and then fine tuned on the target dataset.
* Used when the target dataset is relatively small in size.
* A real world example can be, using your knowledge of riding a bicycle to learn how to ride a motorcycle.

## Dataset
#### IMDb Dataset
* Dataset can be downloaded from [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/).
* Pre-processed dataset can be downloaded from reporsitory [here](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Dataset/IMDB_prePro.csv).
* 25000 highly polar movie reviews for training
* 25000 movie reviews for testing
* Classes of the dataset balanced, so no class biasness observed.

## Pre-Processing
* As discussed in the starter project, Pre-Processing is an important step for text data to make the text more understandable.
* Complete script can be found [here](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Notebook/pre-processing.ipynb)
* Tweet tokenizer from the NLTK Library is used as follows:
```python
tknzr = TweetTokenizer(reduce_len=True, preserve_case=False, strip_handles=False)
```
* The following function substitutes chunk of texts that follow the style as specified in the ```pattern``` variable with the one specified in ```repl``` variable.
```python
def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)
```
* Label sentiment texts are converted to numbers using:
```python
encode_label = {'negative' : 0, 'positive' : 1}
df['sentiment'] = df['sentiment'].map(encode_label)
```
* Final pre-processed dataframe is saved as csv file using:
```python
df.to_csv(FILE OUTPUT PATH)
```

## Implementing Transfer Learning

### Model Architecture
* Before discussing the steps of Transfer Learning, we give a brief introduction of our model.
* We use a customized RoBERTa, which contains the RoBERTa model with some additional layers in the end.
* The model definition contains two parts, the first contains description of layers, the second contains the order of layers.
* The layers used are:
```python
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 256)
        self.bn1 = torch.nn.LayerNorm(256)
        self.l2 = torch.nn.Linear(256, 64)
        self.bn2 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l3 = torch.nn.Linear(64, 2)
```
* ```self.roberta``` contains the layers of the RoBERTa model.
* ```self.d1``` and ```self.d2``` drops out random pecentage of neurons from the incoming layers during training. The percentage for dropout is defined using ```dropout_rate```, and since the dropout is completely random the model becomes robust.
* ```self.l1```, ```self.l2```, and ```self.l3``` are Linear layers. The input parameters for Linear layers consist of input neurons or incoming neurons and output neurons.
* ```self.l1``` has 768 incoming neurons as RoBERTa model has output embedding of 768 units in the final layer. ```self.l1``` gives 256 output units.
* Similarly ```self.l2``` has 256 input neurons and outputs 64 units or neurons.
* ```self.l3``` is our final layer having 2 output neurons. These 2 output neurons decide the class of the input sentence to the model.
* ```self.bn1``` and ```self.bn2``` are [Normalization layers](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html). They normalize the output of the layers and ensure that the value lies between 0 and 1.
* Once we have our layers defined, we need to tell the model how to use them to compute the outputs. The order in which the layers occur is shown in the diagram below.
* The above flowchart can be implemented using:
```python
def forward(self, input_ids, attention_mask):
        _, x = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = self.d1(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = self.l2(x)
        x = self.bn2(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)
        x = self.l3(x)
        
        return x
```
* Now that our model is defined, we can start with the implementation.

### Pre-Training
* Pre-training involves the beginner learning phase of the model.
* Model is usually trained on a large dataset in this step for a higher number of epochs.
* Can be referred as the initial learning phase where weights are assigned.
* The first step for this step is to assign the hyperparameters. This is done using:
```python
MAX_SEQ_LEN = 256
```
* MAX_SEQ_LEN defines the maximum lenght of text to be considered.
```python
PRE_TRAINING_TRAIN_BATCH_SIZE = 32
```
* PRE_TRAINING_TRAIN_BATCH_SIZE defines number of examples after which weights are updated using the loss function.
```python
PRE_TRAINING_VAL_BATCH_SIZE = 64
PRE_TRAINING_TEST_BATCH_SIZE = 64
```
* PRE_TRAINING_VAL_BATCH_SIZE & PRE_TRAINING_TEST_BATCH_SIZE are number of examples in a batch for validation and testing, no weights are updated during this phase.
```python
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
PRE_TRAINING_DATASET_PATH = "./IMDB_prePro1.csv"
```
* PAD_INDEX & UNK_INDEX take care of shorter text by padding and unknown vocabulary by assigning constant index respectively.
* PRE_TRAINING_DATASET_PATH holds the path of the pre-processed dataset, this is the only value you need to change as per the location of your dataset.
* Now that the hyperparameters are set, we load the model using the [Field function](https://pytorch.org/text/data.html#field).
* We define two different fields, one for text and one for label. This is done using:
```python
# Define columns to read.
review_field = Field(use_vocab=False, 
                   tokenize=tokenizer.encode, 
                   include_lengths=False, 
                   batch_first=True,
                   fix_length=MAX_SEQ_LEN, 
                   pad_token=PAD_INDEX, 
                   unk_token=UNK_INDEX)
label_field = Field(sequential=False, use_vocab=False, batch_first=True)

fields = {'review' : ('review', review_field), 'label' : ('label', label_field)}
```
* Once the fields are created, we turn them into tabular dataset just like excel files, this is done using the [TabularDataset](https://pytorch.org/text/data.html#tabulardataset) function. The dataset is created using:
```python
train, valid, test = TabularDataset(path=PRE_TRAINING_DATASET_PATH, 
                                                   format='CSV', 
                                                   fields=fields, 
                                                   skip_header=False).split(split_ratio=[0.70, 0.1, 0.2], 
                                                                            stratified=True, 
                                                                            strata_field='label')

```
* You can change the ratio of train, validation and test set by changing the values in ```split_ratio```.
* We now define the iterators which load the datasets during training, validation and testing.
```python
training_set_iter = Iterator(train, batch_size=PRE_TRAINING_TRAIN_BATCH_SIZE, device=device, train=True, shuffle=True, sort=False)
valid_set_iter = Iterator(valid, batch_size=PRE_TRAINING_VAL_BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)
test_set_iter = Iterator(test, batch_size=PRE_TRAINING_TEST_BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)
```
* Once everything is set, we can start with our pre-training. But before that we need to make a few lists that would hold the training loss, validation loss, and . These will be used to observe the performance of the model in later section.
* The empty lists are defined as:
```python
train_loss_list = []
val_loss_list = []
epc_list = []
```
* In our code, the ```pretrain((model, optimizer, training_set_iter, valid_set_iter, scheduler, num_epochs):``` function contains the pre-training steps.
* Here we shall explain in bits what each line does.
* For our model, we have decided not to train the RoBERTa model, to do this we need to freeze the weights. This is done using:
```python
for param in model.roberta.parameters():
        param.requires_grad = False
    
    model.train()
```
* We then define our loss function. In case you still are facing difficulties understanding these terms, please go through our [starter project]().
* The loss function we have used is [Cross Entropy Loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
```python
criterion = torch.nn.CrossEntropyLoss()
```
* Rather than saving the weights at every step, we only save the weights which give the least validation loss. The flowchart below explain the process.

* We define the best_valid_loss as infinity in the starting to implement our criteria for saving the model with least validation loss.
```python
best_valid_loss = float('Inf')
```
* We then progress through the epochs using a for loop defined as:
```python
 for epoch in range(num_epochs):
 ```
 * At the beginning of each epoch, the train and validation loss are set to 0.0.
 ```python
train_loss = 0.0
valid_loss = 0.0
 ```
    
    
    
   
         
        for (review, label), _ in training_set_iter:
            mask = (review != PAD_INDEX).type(torch.uint8)
            y_pred = model(input_ids=review, attention_mask=mask)
            loss = criterion(y_pred, label)
            loss.backward()
            optimizer.step()    
            scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()
                
        model.eval()
        with torch.no_grad():                    
            for (review, target), _ in valid_set_iter:
                mask = (review != PAD_INDEX).type(torch.uint8)
                y_pred = model(input_ids=review, attention_mask=mask)
                loss = criterion(y_pred, target)
                valid_loss += loss.item()

        train_loss = train_loss / len(training_set_iter)
        valid_loss = valid_loss / len(valid_set_iter)
        
        train_loss_list.append(train_loss)
        val_loss_list.append(valid_loss)
        epc_list.append(epoch)

        # print summary
        print('Epoch [{}/{}], Pre-Training Loss: {:.4f}, Val Loss: {:.4f}'
              .format(epoch+1, num_epochs, train_loss, valid_loss))
        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss 
            # Saving Best Pre-Trained Model as .pth file
            torch.save({'model_state_dict': model.state_dict()}, "./best_pre_train_model.pth")
    
    # Set bert parameters back to trainable
    for param in model.roberta.parameters():
        param.requires_grad = True
     
    
        
    print('Pre-training done!')
```
Now let’s look at the code and the explanation. As you can see, we have set  
```python
param.requires_grad = False
```
in the pre-training step. This is to ensure that during the pre-training the gradients of the RoBERTa model are not changed or updated, only the layers which follow the original pre-trained model are trained as per the IMDb dataset. The criterion/ loss function is set to Cross Entropy Loss and we have set the best validation loss to infinity. The use of this best validation loss will be explained further. The train loop begins and we set the train loss and valid loss to zero each. We then use the train iterator to load the first training batch and calculate the predictions using
```python
y_pred = model(input_ids=source, attention_mask=mask)
```
Then the losses are calculated which take the predicted values and the original values related to each example in the training batch. The ```loss.backward()``` function initiates the backpropagation and the weights are updated. We have also used a ```scheduler.step()``` to keep track of our learning rate. We have used a decaying learning rate to decrease the learning rate as the training progresses, this helps in overcoming overfitting. We then store the training losses, and increase the number of global steps by 1. At the end we check if the global step is equal to our validation period, to test the model on our validation data. Before we proceed with our validation steps, we freeze the gradients of our model using
```python
with torch.no_grad():
```
As we don’t want our model to update the gradients during the validation period. We then calculate the predictions and calculate the validation loss. We then calculate this validation loss to our best validation loss, in case our validation loss comes out to be less, the model is saved and the training continues as usual. If the loss is more, the model is not saved and the training continues as usual. This makes sure that we only save those checkpoints, where the model best fits our validation dataset, helping us to curb overfitting. Once the training is complete, we set the training parameters of RoBERTa back to true. This was the explanation of the pretrain function. The next piece of code, sets the hyperparameters and calls the function to initiate the pre-training. The model is run for 12 epochs, and the model having best validation accuracy is saved. In the next section we will use this saved model to create our classifier and test its performance.  
The model can be evaluated using the script given below. 
```python
def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (source, target), _ in test_loader:
                mask = (source != PAD_INDEX).type(torch.uint8)
                
                output = model(source, attention_mask=mask)

                y_pred.extend(torch.argmax(output, axis=-1).tolist())
                y_true.extend(target.tolist())
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax = plt.subplot()

    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['negative', 'positive'])
    ax.yaxis.set_ticklabels(['negative', 'positive'])
```
Now that we have the means to evaluate our model, let us initiate the pre-training. This can be done using
```python
PRE_TRAINING_NUM_EPOCHS = 12
steps_per_epoch = len(training_set_iter)

PRE_TRAINING_model = ROBERTA(0.4)
PRE_TRAINING_model = PRE_TRAINING_model.to(device)


PRE_TRAINING_optimizer = AdamW(PRE_TRAINING_model.parameters(), lr=1e-4)
PRE_TRAINING_scheduler = get_linear_schedule_with_warmup(PRE_TRAINING_optimizer, 
                                            num_warmup_steps=steps_per_epoch*1, 
                                            num_training_steps=steps_per_epoch*PRE_TRAINING_NUM_EPOCHS)
print('Pre-training starts')
pretrain(model=PRE_TRAINING_model, training_set_iter=training_set_iter, valid_set_iter=valid_set_iter, optimizer=PRE_TRAINING_optimizer, scheduler=PRE_TRAINING_scheduler, num_epochs=PRE_TRAINING_NUM_EPOCHS)
```
The above code generates the output shown below  
<p align="center">
  <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/PreTrainingStats.png" />
</p> 
The graph below shows the change in Training set and Validation losses as the training of the model progresses.
<p align="center">
  <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/TrainValidLossGraph.png" />
</p>
As seen from the graph above, the validation set loss is lower than the training set loss. This occurs due to our Dropout layer. The dropout layer randomly drops a fraction of neurons during the training, which leads to a decrease in accuracy, but this makes the model more robust, as now the model would perform much better when no nodes are dropped during validation and testing. The image below contains the confusion matrix generated during the pre-training. 
You can checkout evaluation code in jupyter notebook roberta.ipynb posted in this GitHub repo. 
<p align="center">
  <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/ConfusionMatrix.png" />
</p> 

The respecctive F1 scores for the classes are 0.8036 and 0.8072. And the ratio of mislabel is almost balenced, hence there are no chances of the model being biased in any way.   

## Creating Classifier with Transfer Learning
### Transfer Learning
When dealing with many classification or segmentation problems in deep learning, the available dataset might not be at par with the problem. One of the major problems is the availability of very limited datasets. There exist many solutions to deal with these problems, Transfer learning is one of them. So, the very first question is what is transfer learning? Transfer learning is a research problem in machine learning that focuses on storing the knowledge a model gains while solving a problem and then use that knowledge to solve a similar problem. For example, a model which already knows how to differentiate between the sentiment of reviews can be used to differentiate between sentiments overall with just a little buff. This buff is known are re-training the model. In other words, we first train the model on a large dataset available for a similar task as our problem, save that knowledge, use that knowledge and train the model on a part of our task’s dataset and then test its performance. Transfer learning has shown great advancements in the field of cancer subtype discovery, building utilization, general game playing, text classification and other tasks. 
In our tutorial, we take a Pre-trained RoBERTa model, transfer that learning to the IMDb dataset, and then further transfer that learning to IMDb dataset again with which we were able to get better result then the previous one. Now, lets us begin with the training of our classifer model.

We first start with setting up the hyperparameters just like we did for the pre-training part. These hyperparameters and their values remain the same. Since the dataset remians the same in this step, the steps remain the same, with only change in variable name as follows
```python
CLASSIFIER_MAX_SEQ_LEN = 256
CLASSIFIER_TRAIN_BATCH_SIZE = 32
CLASSIFIER_VAL_BATCH_SIZE = 64
CLASSIFIER_TEST_BATCH_SIZE = 64
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
CLASSIFIER_DATASET_PATH = "./final_prepro1.csv"

review_field = Field(use_vocab=False, 
                   tokenize=tokenizer.encode, 
                   include_lengths=False, 
                   batch_first=True,
                   fix_length=CLASSIFIER_MAX_SEQ_LEN, 
                   pad_token=PAD_INDEX, 
                   unk_token=UNK_INDEX)

label_field = Field(sequential=False, use_vocab=False, batch_first=True)

fields = {'review' : ('review', review_field), 'label' : ('label', label_field)}


train, valid, test = TabularDataset(path=CLASSIFIER_DATASET_PATH, 
                                                   format='CSV', 
                                                   fields=fields, 
                                                   skip_header=False).split(split_ratio=[0.70, 0.1, 0.2], 
                                                                            stratified=True, 
                                                                            strata_field='label')

training_set_iterC = Iterator(train, batch_size=CLASSIFIER_TRAIN_BATCH_SIZE, device=device, train=True, shuffle=True, sort=False)
valid_set_iterC = Iterator(valid, batch_size=CLASSIFIER_VAL_BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)
test_set_iterC = Iterator(test, batch_size=CLASSIFIER_TEST_BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)
```
We then create the exact same replica of our customized RoBERTa model. In the next piece of code, we create lists to store the training loss, validation loss and global steps. These shall be used to plot the trends in losses during the training period. Then we initiate the classifier, only to use the ```torch.load()``` to load the saved model instead of a random initiation. This is known as creating the transfer model. Since the architectures of both the models are same, we don’t need to make any changes to the load function. Once again, we set the best validation loss to infinity and the loss function to Cross Entropy Loss. The epochs are started similarly, and the training progresses. Also, it must be noticed that, this time we have not frozen the RoBERTa layers, and allowed them to train as well. This ensures that the model fits itself to the dataset well. The rest of the training procedure is exactly same as that of the pre-training method. We freeze the weights during the testing of validation set, and store both the training and testing losses in our lists. The model having least validation loss is saved and then can be deployed on the Web Application, which we will discuss in the further sections of our tutorial. The code for this part remains exaclty same as for that of the Pre-training hence it is not mentioned again.  
The classifier training is initiated using  
```python

CLASSIFIER_NUM_EPOCHS = 20
steps_per_epoch = len(training_set_iter)

CLASSIFIER_model = ROBERTA()
CLASSIFIER_model = CLASSIFIER_model.to(device)
preTrained = torch.load("./best_pre_train_model.pth")
CLASSIFIER_model.load_state_dict(preTrained, strict=False)

CLASSIFIER_optimizer = AdamW(CLASSIFIER_model.parameters(), lr=1e-4)
CLASSIFIER_scheduler = get_linear_schedule_with_warmup(CLASSIFIER_optimizer, 
                                            num_warmup_steps=steps_per_epoch*1, 
                                            num_training_steps=steps_per_epoch*CLASSIFIER_NUM_EPOCHS)
print('Training starts')
classifier(model=CLASSIFIER_model,optimizer=CLASSIFIER_optimizer, training_set_iter=training_set_iter, valid_set_iter=valid_set_iter,  scheduler=CLASSIFIER_scheduler, num_epochs=CLASSIFIER_NUM_EPOCHS)
```
So, the model runs for 20 epochs and the outcome is generated as  
<p align="center">
  <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/FinalTrainStats.png" />
</p> 
In the last piece of our code, we use the final model for the test set. We use the ```torch.no_grad()``` to freeze the weights again, and test the model. Once the model is trained, we can compare it with the previous model, to evaluate if the model has improved or degraded in terms of classifying quality.  
The image below shows the trends in Training and Validation set losses.  
<p align="center">
  <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/TrainValidLossGraphFinalTraining.png" />
</p> 

As noticed earlier, the losses for validation set is still lower as compared to that of training set due to our Dropout layer. The confusion matrix below gives us a comparison for the test set.  
<p align="center">
  <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/ConfusionMatrixFinal.png" />
</p>   
As compared from the Pre-train confusion matrix, the final classifier has an improved F1 score of 0.8108 and 0.8136, hence the misclassfication error has also reduced. Hence concluding that, transfer learning helps to increase the performance when used for similar datasets.  
In the end we save the model using `torch.save(model.state_dict(), "modelName.pth")` so that it can be deployed to our web application.


## Deploy Deep Learning Model using Flask
Now that our classification model is ready and saved, we can deploy it using a simple Flask application. Before we start with the code, let us see what Flask is. Flask is a simple web application framework for Python, which allows the user to write applications without worrying about protocol or thread management. You can learn more about Flask from their official documentation [here](https://flask.palletsprojects.com/en/1.1.x/).  
```python
state_dict = torch.load(cwd + '/ml_model/modelFinal.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)


def pred(text):
    print("Text Received =>", text)
    word_seq = np.array([vocab[word] for word in text.split() 
                      if word in vocab.keys()])
    word_seq = np.expand_dims(word_seq,axis=0)
    t = torch.from_numpy(word_seq).to(torch.int64)
    length = torch.LongTensor([1])
    output = model(t, length)
    print("Got output - ",output)
    pro = (output.item())
    status = "positive" if pro < 0.5 else "negative"
    return status
```
The above code snippet is our predict function which takes the input text and then proceeds to use the model to predict the sentiment. We first convert our input text into the vocabulary input and then convert these into PyTorch tensors. The tensors are then given to the model, which returns the ```output```.  
In our Flask application, we add the saved model, and a [predict.py](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Webapp/Flask/ml_model/predict.py) file that makes the predictions. This pred.py file contains the architecture of the model, that would load the saved weights of the model. Since, we had space limitations for deployment, we used a simple LSTM model to demonstrate this part. The code remains exactly same for deploying RoBERTa, only the class ```LSTM()``` would be replaced by our RoBERTa model. This model that we have used is explained in the end for your understanding. In the pred.py file, we create the model and then load the saved weights using `model.load_state_dict()` function, then we have a function ```pred()``` that uses the model to predict the incoming review request. The front-end sends the review using a form which has been created using Prediction component which will be explained in the next section. For now let us just think of it as simple input parameter sent by the front-end application. The model then predicts the sentiment and returns the sentiment. The app.py file contains the encapsulation of our request and post API. We use the POST API to send the final sentiment of the tweet to our front-end. Now that our back-end is complete, we can proceed with our Front-end application. The description of our front-end application is mentioned in the next section.  

## Creating a React Front-End
We have created a very simple React front-end for our tutorial. To initiate the process, we first create an empty folder and begin with the command
```js
npx create-react-app applicationName
```
This would download the required libraries to the folder and you can initiate the basic application using
```js
cd my-app
npm start
```
The next step would be to create the required components for our application. Before we get into the details of components, let us see what components in React actually mean. As per the official React documentation, components are similar to Javascript functions, which take arbitrary inputs and return React elements which describe what should appear on the screen. In simple language they are pieces of code, that can be used repeatedly and exist independently. More details about components can be found [here](https://reactjs.org/docs/react-component.html). We create the components in the in the [componenets folder](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/tree/main/Webapp/React/src/components). We have created two components named as Example and Prediction. The Example component uses a simple Render function to display the examples, while the Prediction component uses a Form group.  
The [example.js](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Webapp/React/src/components/example.js) component contains the details of the examples visible on our webpage, while the [prediction.js](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Webapp/React/src/components/prediction.js) component contains a form, which allows us to submit a sentence for testing the sentiment. This testing of the sentence is executed using our custom created API, which sends a request to the server containing the sentence. The server receives the request, tests the sentence using the classifier and then returns the sentiment using get request. The visuals of the project can be changed by making changes in the from the [index.css](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Webapp/React/src/index.css) file.  

## LSTM Model used in Web Application
Since, Heroku only allows a total space of 500Mb to be uploaded, we could not host our final RoBERTa model there due to its humongous size. To demonstrate the working of our web application we decided to proceed with a lighter LSTM model. You can find the architecture of the model in [predict.py](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Webapp/Flask/ml_model/predict.py) file. The model is defined in the class ```LSTM()```. The model has an embedding layer of size 500, which takes the sentences and returns the vocabulary embedding of the words. Then we have a bidirectional LSTM layer having 128 units, the bidirectional nature of the layer allows the model to look at the sentence in both front and back order. We then have a Dropout layer which drops 30% of the words LSTM output units during training. This makes our model more robust and would increase the accuracy during testing as explained above. The final layer is a Linear layer having single output. If the output is smaller than 0.5, we annotate it to the negative class, while having a value greater than 0.5 means the sentence belongs to the positive class. This brings us to  the end of our tutorial.

## Summarizing it all
The project is finally complete. To summarize it, here is the encapsulated version of it all. We first created a pre-trained model of our choice, which was transferred to an exact replica, the replica was trained on a similar dataset (same dataset) in our tutorial. This model was then saved for deployment. A flask application was created which had functions to handle the post and get requests from the front-end. The flask application has a prediction.py file which loads the saved model and uses it to make predictions. Then we created a front-end using React, where the user can enter reviews through a form and get the desired output sentiment. The demonstration for the latter part was done using a simple LSTM model due to limitations in space. The final application was then hosted on Heroku server.


