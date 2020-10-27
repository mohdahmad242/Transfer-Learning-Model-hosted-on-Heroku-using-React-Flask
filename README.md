# Transfer Learning Model hosted on Heroku using React & Flask
> Transfer Learning model using RoBERTa on IMDb dataset deployed on React and Flask.  
> Try out [here](https://imdbmovienew.herokuapp.com/), Run on Google Collab [here](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Notebook/roberta.ipynb)
## Introduction
Today, we are going to teach you a way to create your own text classifier which can be used to check whether the sentiment a sentence carries is positive or negative. The implementation of this deep learning model is based on RoBERTa, which is a Robustly Optimized BERT Pretraining approach. This pre-trained RoBERTa model is first pre-trained on the [IMDb dataset](#IMDb-Dataset), after which the learning was transferred to the IMDb dataset again. To make the model presentable to a user, we will be deploying it on a front-end application created using ReactJS. You should have basic knowledge of Deep Learning in pytorch. The contents of this tutorial are given as follows:
1. [Pre-training](#Pre-training-Customized-RoBERTa)  
    - Download the pre-trained RoBERTa model  
    - Set up the IMDb dataset into the input format of RoBERTa  
    - Use the IMDb dataset to pre-train the model  
    - Save the model’s weight to use for transfer learning
2. [Use the saved weights to implement Transfer Learning](#Creating-Classifier-with-Transfer-Learning)  
    - Use the saved weights from Step 1 and recreate the RoBERTa model
    - Set up the IMDb into the input format of RoBERTa
    - Use the recreated model and train it using the IMDb dataset changing the final layer only
    - Test the model using the reserved test set and save the results  
3. [Deploy the model on Heroku server](#Deploy-Deep-Learning-Model-using-Flask)  
    - Create a Flask backend
    - Create a React front-end
    - Synchronize the front-end and back-end
    - Deploy the project using Heroku  
The whole tutorial is broken down into these 3 sections to make it easier and simpler to understand. The first section is the Pre-training of our RoBERTa model. But before that we would throw some light on the dataset we have used.  

## Dataset
#### IMDb Dataset
The IMDb dataset is made available at [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/), it has 25000 highly polar movie reviews for training and other 25000 for testing. The class distribution of these reviews is almost equal, i.e. the number of positive reviews is approximately equal to the number of negative reviews, hence no class biasness would occur during the pre-training of the model. We took the complete 50000 tweets of the dataset and divided them in the ratio of 8:1:1 for training, validation and testing. We have used a pre-processed version of the dataset, which can be downloaded from our repository from [here](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Dataset/IMDB_prePro.csv), you can also view the pre-processing script available [here](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Notebook/pre-processing.ipynb)  


## Pre-training Customized RoBERTa
The pre-training steps are quite simple to understand. The code given below would manipulate the data for input into the model.  
```
MAX_SEQ_LEN = 256
PRE_TRAINING_TRAIN_BATCH_SIZE = 32
PRE_TRAINING_VAL_BATCH_SIZE = 64
PRE_TRAINING_TEST_BATCH_SIZE = 64
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
PRE_TRAINING_DATASET_PATH = "./IMDB_prePro1.csv"

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


train, valid, test = TabularDataset(path=PRE_TRAINING_DATASET_PATH, 
                                                   format='CSV', 
                                                   fields=fields, 
                                                   skip_header=False).split(split_ratio=[0.70, 0.1, 0.2], 
                                                                            stratified=True, 
                                                                            strata_field='label')

training_set_iter = Iterator(train, batch_size=PRE_TRAINING_TRAIN_BATCH_SIZE, device=device, train=True, shuffle=True, sort=False)
valid_set_iter = Iterator(valid, batch_size=PRE_TRAINING_VAL_BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)
test_set_iter = Iterator(test, batch_size=PRE_TRAINING_TEST_BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)
```
The various terms that have been used in the code above are explained as follow:
- **MAX_SEQ_LEN**: This hyperparameter is used to define the maximum length of a sentence which will be taken into consideration. That is, the maximum number of words from each review which will be provided to the model.
- **PRE_TRAINING_TRAIN_BATCH_SIZE**: The number of reviews after which the model updates its weights. In simple terms, it is the number of reviews after which loss is calculated and weights are revised on the basis of loss. A smaller training batch size means that the weights are updated more frequently.
- **PRE_TRAINING_VAL_BATCH_SIZE**: The number of reviews which will be given to the model during the validation of the model. During the validation period, the weights of the model are frozen to check its performance against a completely new dataset.
- **PAD_INDEX**: As discussed in the MAX_SEQ_LEN, all reviews must have the same length when they are fed into the model. While the longer reviews are trimmed, the shorter ones are padded by adding 0 either at the beginning or at the end of the sentence. This task is done by PAD_INDEX.
- **UNK_INDEX**: It might be possible that not all words present in the reviews might be available in the vocabulary files, this can be due to multi-lingual reviews, authors making typos while posting the reviews or many such reasons. The UNK_INDEX is used to handle these unknown or out-of-vocabulary words by changing all them to zero.
- **Field**: Field is a pre-defined class model in Pytorch which is used to create a datatype that converts sentences or words to tensors. It contains the instructions like how to tokenize the sentences, maximum sentence length to be taken into consideration, type of padding, how to handle unknown vocabulary, etc. More information about field can be found at the official documentation of Pytorch [here](https://pytorch.org/text/data.html#field).
- **TabularDataset**: TabularDataset creates a dataset in either CSV, TSV or JSON format given the path of the file and the Fields to be considered. We have used the CSV format with a 0.8, 0.1, 0.1 split for train, test and validation dataset for the IMDb dataset, and for the SST2 dataset the ratios were set to 0.7, 0.1 and 0.2. To know more about TabularDataset, try the Pytorch official documentation [here](https://pytorch.org/text/data.html#tabulardataset).
- **Iterator**: As discussed in the PRE_TRAINING_TRAIN_BATCH_SIZE, the training dataset is divided into batches before being fed to the network. This division of batches of the training, validation or testing data is performed by the Iterator function which loads the batches from the provided dataset.
This concludes our discussion for the important terminologies which will be used repeatedly. Now let us continue with the model.  

## Model Architecture
The code snipped below mentions our customised RoBERTa model, encapsulated in a class.
```
class ROBERTA(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ROBERTA, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 256)
        self.bn1 = torch.nn.LayerNorm(256)
        self.l2 = torch.nn.Linear(256, 64)
        self.bn2 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l3 = torch.nn.Linear(64, 2)
        
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
The original implementation would look something like:
```
train_loss_list = []
val_loss_list = []
epc_list = []

def pretrain(model, optimizer, training_set_iter, valid_set_iter, scheduler, num_epochs):
    
    # Pretrain linear layers, do not train bert
    for param in model.roberta.parameters():
        param.requires_grad = False
    
    model.train()
    
    criterion = torch.nn.CrossEntropyLoss()
    best_valid_loss = float('Inf')
    
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0 
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
The model can be evaluated using the script given below. 
```
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
```
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
```
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
```

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
```
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
```
npx create-react-app applicationName
```
This would download the required libraries to the folder and you can initiate the basic application using
```
cd my-app
npm start
```
The next step would be to create the required components for our application. Before we get into the details of components, let us see what components in React actually mean. As per the official React documentation, components are similar to Javascript functions, which take arbitrary inputs and return React elements which describe what should appear on the screen. In simple language they are pieces of code, that can be used repeatedly and exist independently. More details about components can be found [here](https://reactjs.org/docs/react-component.html). We create the components in the in the [componenets folder](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/tree/main/Webapp/React/src/components). We have created two components named as Example and Prediction. The Example component uses a simple Render function to display the examples, while the Prediction component uses a Form group.  
The [example.js](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Webapp/React/src/components/example.js) component contains the details of the examples visible on our webpage, while the [prediction.js](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Webapp/React/src/components/prediction.js) component contains a form, which allows us to submit a sentence for testing the sentiment. This testing of the sentence is executed using our custom created API, which sends a request to the server containing the sentence. The server receives the request, tests the sentence using the classifier and then returns the sentiment using get request. The visuals of the project can be changed by making changes in the from the [index.css](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Webapp/React/src/index.css) file.  

## LSTM Model used in Web Application
Since, Heroku only allows a total space of 500Mb to be uploaded, we could not host our final RoBERTa model there due to its humongous size. To demonstrate the working of our web application we decided to proceed with a lighter LSTM model. You can find the architecture of the model in [predict.py](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Webapp/Flask/ml_model/predict.py) file. The model is defined in the class ```LSTM()```. The model has an embedding layer of size 500, which takes the sentences and returns the vocabulary embedding of the words. Then we have a bidirectional LSTM layer having 128 units, the bidirectional nature of the layer allows the model to look at the sentence in both front and back order. We then have a Dropout layer which drops 30% of the words LSTM output units during training. This makes our model more robust and would increase the accuracy during testing as explained above. The final layer is a Linear layer having single output. If the output is smaller than 0.5, we annotate it to the negative class, while having a value greater than 0.5 means the sentence belongs to the positive class. This brings us to  the end of our tutorial.

## Summarizing it all
The project is finally complete. To summarize it, here is the encapsulated version of it all. We first created a pre-trained model of our choice, which was transferred to an exact replica, the replica was trained on a similar dataset (same dataset) in our tutorial. This model was then saved for deployment. A flask application was created which had functions to handle the post and get requests from the front-end. The flask application has a prediction.py file which loads the saved model and uses it to make predictions. Then we created a front-end using React, where the user can enter reviews through a form and get the desired output sentiment. The demonstration for the latter part was done using a simple LSTM model due to limitations in space. The final application was then hosted on Heroku server.


