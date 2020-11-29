# Transfer Learning Model hosted on Heroku using React & Flask
> Transfer Learning model using RoBERTa on IMDb dataset deployed on React and Flask.  
> Try out [here](https://imdbmovienew.herokuapp.com/), Run on Google Collab [here](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Notebook/roberta.ipynb)

<br>
<h1 align="center">Chapter One</h1>
<h2 align="center">Training Deep Learning Models</h2>
<p align="center">
 <a href="https://circleci.com/gh/huggingface/transformers">
  <img alt="Build" src="https://img.shields.io/badge/python-3%2B-brightgreen?logo=Python">
 </a>
 <a href="https://circleci.com/gh/huggingface/transformers">
  <img alt="Build" src="http://img.shields.io/static/v1?label=Pytorch&message=1.6.0&color=brightgreen&logo=Pytorch">
 </a>
</p>
<br>

## Overview
 
<p align="center">
  <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/finalResult.gif">
</p>  

In this tutorial we will create a text classifier using RoBERTa model in PyTorch and deploy it using Heroku on a web application created using Flask and React JS. We perform the task of [Sentiment Analysis](https://towardsdatascience.com/sentiment-analysis-concept-analysis-and-applications-6c94d6f58c17) on a given piece of text, and try to classify the text as either positive or negative. The topics covered in our tutorial are:
* Chapter 1: [Creating a Text Classifier]()
* Chapter 2: [Deployment of Machine Learning Model on Heroku using Flask](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Webapp/Flask/README.md)
* Chapter 3: []() 

## Pre-Requisities for the Complete Tutorial
To implement the complete project you will need the following:
* Create a [GitHub account](https://github.com/join)
* Create a [Heroku account](https://signup.heroku.com/)
* Have knowledge of PyTorch and Deep Learning, follow the starter project [here](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Notebook/starter%20model.ipynb)

## Introduction
[PyTorch](https://github.com/pytorch/pytorch) is a Python package based on [Torch](http://torch.ch/). PyTorch is one of the most used Deep Learning libraries in recent times for the following reasons:
+ It provides Tensor computation with strong GPU acceleration.
+ Unlike other Deep Learning Libraries where we first define the entire computation graph before running the model, PyTorch allows dymanic defining of graphs.
+ It works excellently well with most used libraries like NumPy, SciPy, and Cython.  


In our tutorial we discuss how to implement [Transfer Learning](https://ruder.io/transfer-learning/index.html#applicationsoftransferlearning) using the PyTorch library and then deploy the model on a web application. By the end of this section, you will be able to execute your own Neural Network using PyTorch and save the weights for reusing. Before we begin let us have a look at the pre-requisites for the section. .

## Pre-Requisites for this section
To implement the project you will need the following.
* Python 3+ installed https://www.python.org/
* Basic Python programming knowledge https://docs.python.org/3/tutorial/
* In case you have a GPU installed in your machine and CUDA enabled you can train it on your local machine 
* Alternatively you can try [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) which provides you open access to GPUs for training a neural network.
* You can also go through our Colab notebook for the project [here]().

### Step 1- Installing Important Libraries
Before initiating this section, we will create a seperate `python environment` for our project. In case you are using Anaconda, open Anaconda Prompt, or else open your Command Prompt in Windows. Linux users can open their terminal instead. Enter the following code to create an environment named as `venv`.
```python
python3 -m venv env
```
* To activate the created environment, enter the following code in your terminal.
```python
source venv/bin/activate
```
* We now install the needed libraries for the project. The below mentioned steps are for poeple using Windows machine. For people using other machine please refer to the installation section [here](https://pytorch.org/) In case you have a GPU with CUDA 10.2 enabled, enter the section 1, else proceed with section 2.
* Section 1:
```python
pip install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install numpy pandas transformer pickel
```
* Section 2:
```python
pip install pytorch torchvision torchaudio cpuonly -c pytorch
pip install numpy pandas transformer pickel
```


### Transfer Learning
Any Deep Learning model works on feature extraction from the given data using the model developed for the task. At the start of any model, the weights of the layers are initiated randomly and then through training iterations the layers learn to extract features from the data. Training these layers takes a lot of time and computation power which is not accessible to everyone, this is where Deep Learning comes into play. Transfer learning focuses on using the knowledge from previous training and implements in on a similar task. It extracts features from a relevant large dataset and then fine-tunes of the given smaller dataset. Below are the reasons why Transfer Learning is a suited training method:
* Acts as an optimization technique that allows rapid progress and improved performance in lesser training time.
* Uses vast knowledge accumulated over a vast training resource for a smaller but related task.
* A real world example can be, using your knowledge of riding a bicycle to learn how to ride a motorcycle.  

In case you want to know more about transfer learning, here are a few resources:
* [What is being transferred in transfer learning ?](https://arxiv.org/abs/2008.11687)
* [A Survey on Deep Transfer Learning](https://arxiv.org/abs/1808.01974)
* [A Gentle Introduction to Transfer Learning for Deep Learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)

## Dataset
#### IMDb Dataset
The IMDb Dataset is a collection of 50000 movie reviews collected from the IMDb website which belong to two different classes 'positive' and 'negative'. The dataset has 25000 labeled reviews for training, and another 25000 reviews for testing.
* Dataset can be downloaded from [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/).
* Pre-processed dataset can be downloaded from reporsitory [here](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Dataset/IMDB_prePro.csv).
* Classes of the dataset balanced, so no class biasness observed.
* View a brief of the dataset below:
<details>
 <summary>View Dataset</summary>
 
 
 | review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | sentiment |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| One of the other reviewers has mentioned   that after watching just 1 Oz episode you'll be hooked. They are right, as   this is exactly what happened with me.<br /><br />The first thing   that struck me about Oz was its brutality and unflinching scenes of violence,   which set in right from the word GO. Trust me, this is not a show for the   faint hearted or timid. This show pulls no punches with regards to drugs, sex   or violence. Its is hardcore, in the classic use of the word.<br   /><br />It is called OZ as that is the nickname given to the Oswald   Maximum Security State Penitentary. It focuses mainly on Emerald City, an   experimental section of the prison where all the cells have glass fronts and   face inwards, so privacy is not high on the agenda. Em City is home to   many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and   more....so scuffles, death stares, dodgy dealings and shady agreements are   never far away.<br /><br />I would say the main appeal of the   show is due to the fact that it goes where other shows wouldn't dare. Forget   pretty pictures painted for mainstream audiences, forget charm, forget   romance...OZ doesn't mess around. The first episode I ever saw struck me as   so nasty it was surreal, I couldn't say I was ready for it, but as I watched   more, I developed a taste for Oz, and got accustomed to the high levels of   graphic violence. Not just violence, but injustice (crooked guards who'll be   sold out for a nickel, inmates who'll kill on order and get away with it,   well mannered, middle class inmates being turned into prison bitches due to   their lack of street skills or prison experience) Watching Oz, you may become   comfortable with what is uncomfortable viewing....thats if you can get in   touch with your darker side. | positive  |
| A wonderful little production. <br   /><br />The filming technique is very unassuming- very old-time-BBC   fashion and gives a comforting, and sometimes discomforting, sense of realism   to the entire piece. <br /><br />The actors are extremely well   chosen- Michael Sheen not only "has got all the polari" but he has   all the voices down pat too! You can truly see the seamless editing guided by   the references to Williams' diary entries, not only is it well worth the   watching but it is a terrificly written and performed piece. A masterful   production about one of the great master's of comedy and his life. <br   /><br />The realism really comes home with the little things: the   fantasy of the guard which, rather than use the traditional 'dream'   techniques remains solid then disappears. It plays on our knowledge and our   senses, particularly with the scenes concerning Orton and Halliwell and the   sets (particularly of their flat with Halliwell's murals decorating every   surface) are terribly well done.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | positive  |
| I thought this was a wonderful way to   spend time on a too hot summer weekend, sitting in the air conditioned   theater and watching a light-hearted comedy. The plot is simplistic, but the   dialogue is witty and the characters are likable (even the well bread   suspected serial killer). While some may be disappointed when they realize   this is not Match Point 2: Risk Addiction, I thought it was proof that Woody   Allen is still fully in control of the style many of us have grown to   love.<br /><br />This was the most I'd laughed at one of Woody's   comedies in years (dare I say a decade?). While I've never been impressed   with Scarlet Johanson, in this she managed to tone down her "sexy"   image and jumped right into a average, but spirited young woman.<br   /><br />This may not be the crown jewel of his career, but it was   wittier than "Devil Wears Prada" and more interesting than   "Superman" a great comedy to go see with friends.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | positive  |
| Basically there's a family where a little   boy (Jake) thinks there's a zombie in his closet & his parents are   fighting all the time.<br /><br />This movie is slower than a   soap opera... and suddenly, Jake decides to become Rambo and kill the   zombie.<br /><br />OK, first of all when you're going to make a   film you must Decide if its a thriller or a drama! As a drama the movie is   watchable. Parents are divorcing & arguing like in real life. And then we   have Jake with his closet which totally ruins all the film! I expected to see   a BOOGEYMAN similar movie, and instead i watched a drama with some   meaningless thriller spots.<br /><br />3 out of 10 just for the   well playing parents & descent dialogs. As for the shots with Jake: just   ignore them.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | negative  |
| Petter Mattei's "Love in the Time of   Money" is a visually stunning film to watch. Mr. Mattei offers us a   vivid portrait about human relations. This is a movie that seems to be   telling us what money, power and success do to people in the different   situations we encounter. <br /><br />This being a variation on   the Arthur Schnitzler's play about the same theme, the director transfers the   action to the present time New York where all these different characters meet   and connect. Each one is connected in one way, or another to the next person,   but no one seems to know the previous point of contact. Stylishly, the film   has a sophisticated luxurious look. We are taken to see how these people live   and the world they live in their own habitat.<br /><br />The only   thing one gets out of all these souls in the picture is the different stages   of loneliness each one inhabits. A big city is not exactly the best place in   which human relations find sincere fulfillment, as one discerns is the case   with most of the people we encounter.<br /><br />The acting is   good under Mr. Mattei's direction. Steve Buscemi, Rosario Dawson, Carol Kane,   Michael Imperioli, Adrian Grenier, and the rest of the talented cast, make   these characters come alive.<br /><br />We wish Mr. Mattei good   luck and await anxiously for his next work.                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | positive  |
</details>
 

## Pre-Processing
As discussed in the starter project, Pre-Processing is an important step for text data to make the text more understandable. The complete pre-processing script can be found [here](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Notebook/pre-processing.ipynb). Given below is the explanation of every step along with their relevance for pre-processing.
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
* The complete code for this available as a notebook in our repository as mentioned above. You can also view the code below:
<details>
 <summary> View Code </summary>
 
 ```python
 import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from wordsegment import segment, load
from nltk.tokenize import TweetTokenizer

STOPWORDS = set(stopwords.words('english'))
tknzr = TweetTokenizer(reduce_len=True, preserve_case=False, strip_handles=False)
def text_preprocess(text):
    text = str(text)
    FLAGS = re.MULTILINE | re.DOTALL
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    text = " ".join([word for word in str(text).split() if word not in STOPWORDS])

    tokens = tknzr.tokenize(text.lower())
    return " ".join(tokens)
#replace INPUT PATH with the path of your file
df = pd.read_csv("INPUT PATH")
# Encoding negative to 0 and positive to 1
encode_label = {'negative' : 0, 'positive' : 1}
df['sentiment'] = df['sentiment'].map(encode_label)
df['review'] = df['review'].apply(text_preprocess)
#replace FILE PATH with your own
df.to_csv("FILE PATH")
 ```
</details>


## Implementing Transfer Learning
The complete implementation of this section can be found on our Repository [here](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Notebook/roberta.ipynb)
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

<p align="center">
  <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/FinalLayerLayout.jpg" />
</p>



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
* The complete model would look something like the code below.
<details>
 <summary>View Code</summary>
 
 ```python
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
</details>


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
* The complete data preparation section would look like this:
<details>
 <summary>View Code</summary>

```python
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
</details>


* Once everything is set, we can start with our pre-training. But before that we need to make a few lists that would hold the training loss, validation loss, and epoch. These will be used to observe the performance of the model in later section.
* The empty lists are defined as:
```python
train_loss_list = []
val_loss_list = []
epc_list = []
```
* In our code, the ```pretrain((model, optimizer, training_set_iter, valid_set_iter, scheduler, num_epochs):``` function contains the pre-training steps.
* Here we shall explain in bits what each line does.
* For our model, we have decided not to train the RoBERTa layers, to do this we need to freeze the weights. This is done using:
```python
for param in model.roberta.parameters():
        param.requires_grad = False
    
    model.train()
```
* We then define our loss function. In case you still are facing difficulties understanding these terms, please go through our [starter project](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Notebook/starter%20model.ipynb).
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
* For every batch, our iterators define earlier load the batch and the model is trained on that batch.
```python
for (review, label), _ in training_set_iter:
            mask = (review != PAD_INDEX).type(torch.uint8)
```
* The predictions are made using:
```python
y_pred = model(input_ids=review, attention_mask=mask)
```
* We then calculate the losses using the criterion definer earlier in the section.
```python
loss = criterion(y_pred, label)
```
* The next steps are to update the weights, and proceed with our optimizer and learning rate scheduler.
```python
loss.backward()
optimizer.step()    
scheduler.step()
optimizer.zero_grad()
```
* As the loop progresses, batches are loaded and the losses are calculated following which weights are updated.
* Once the epoch is complete, we freeze the weights and calculate the losses for the validation set. This is done using:
```python
with torch.no_grad():                    
            for (review, target), _ in valid_set_iter:
                mask = (review != PAD_INDEX).type(torch.uint8)
                y_pred = model(input_ids=review, attention_mask=mask)
                loss = criterion(y_pred, target)
                valid_loss += loss.item()
```
* We then calculate the aggregate training and validation loss and append them to our list along with the epoch number.
```python
train_loss = train_loss / len(training_set_iter)
        valid_loss = valid_loss / len(valid_set_iter)
        
        train_loss_list.append(train_loss)
        val_loss_list.append(valid_loss)
        epc_list.append(epoch)
```
* We then print the training summary of our epoch using:
```python
print('Epoch [{}/{}], Pre-Training Loss: {:.4f}, Val Loss: {:.4f}'
              .format(epoch+1, num_epochs, train_loss, valid_loss))
```
* As discussed above, we only save the model having lowest validation score, we compare the current validation score with our best score. In case the current validation score is lower, the model is saved using:
```python
if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss 
            # Saving Best Pre-Trained Model as .pth file
            torch.save({'model_state_dict': model.state_dict()}, "./best_pre_train_model.pth")
```
* Once our pre-training is complete we set the weights of RoBERTa layers to trainable again.
```python
for param in model.roberta.parameters():
        param.requires_grad = True
```

* Once completed, the pre-training block would look like:

<details>
 <summary>View Code</summary>

```python
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
</details>


* The above set of codes defined our pre-training steps, now lets look at how the pre-training actually takes place.
* We first set the number of epochs, steps per epoch, and  learning rate using:
```python
PRE_TRAINING_NUM_EPOCHS = 12
steps_per_epoch = len(training_set_iter)

PRE_TRAINING_model = ROBERTA(0.4)
PRE_TRAINING_model = PRE_TRAINING_model.to(device)


PRE_TRAINING_optimizer = AdamW(PRE_TRAINING_model.parameters(), lr=1e-4)
PRE_TRAINING_scheduler = get_linear_schedule_with_warmup(PRE_TRAINING_optimizer, 
                                            num_warmup_steps=steps_per_epoch*1, 
                                            num_training_steps=steps_per_epoch*PRE_TRAINING_NUM_EPOCHS)
```
* We then pass on these paramters to our ```pretrain``` function for the training to begin.
```python
pretrain(model=PRE_TRAINING_model, training_set_iter=training_set_iter, valid_set_iter=valid_set_iter, optimizer=PRE_TRAINING_optimizer, scheduler=PRE_TRAINING_scheduler, num_epochs=PRE_TRAINING_NUM_EPOCHS)
```
* The complete code for this section is given below:
<details>
 <summary>View Code</summary>

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
</details>


* The training summary generated during this phase is shown below:
<p align="center">
  <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/PreTrainingStats.png" />
</p>

* Remember the lists that we had defined earlier, we can use them to visualize the trend of these losses in graphical format.  
```python
plt.figure(figsize=(10, 8))
plt.plot(epc_list, train_loss_list, label='Train')
plt.plot(epc_list, val_loss_list, label='Valid')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=14)
plt.show()
```

* The above code generates the result shown in the image below:
<p align="center">
  <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/TrainValidLossGraph.png" />
</p>

* Now that we have our pre-trained model, we can use it to evaluate our test set which was created earlier. This would tell us how successful our pretraining was.
* We freeze the weights of all layers during our evaluation. The evaluation function has two lists, one contains the predicted values and the other holds the true values.
```python
def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
```
* We then freeze the weights and use the model for predicting the test set.            
```python
with torch.no_grad():
        for (source, target), _ in test_loader:
                mask = (source != PAD_INDEX).type(torch.uint8)
                
                output = model(source, attention_mask=mask)

                y_pred.extend(torch.argmax(output, axis=-1).tolist())
                y_true.extend(target.tolist())
```
* Once the predictions are made, we visualize the result using [Classification Report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html):
```python
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
* The complete evaluation function would look like:

<details><summary>View Code</summary>

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

</details>

* The output obtained is shown in the image below:
<p align="center">
  <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/ConfusionMatrix.png" />
</p> 

* This concludes our discussion for the Pre-Training. We now move forward to implementing Transfer Learning.

### Implementation of Transfer Learning

* We use a replica of the model defined earlier in the above sections.
* The weights of the model having least validation error are loaded into our new replica model.
* This model is then trained on the dataset, and shows an improvement in the result.
* The final weights are saved, which can then be deployed on the web application.
* The first step is to define the hyperparamters again.
```python
CLASSIFIER_MAX_SEQ_LEN = 256
CLASSIFIER_TRAIN_BATCH_SIZE = 32
CLASSIFIER_VAL_BATCH_SIZE = 64
CLASSIFIER_TEST_BATCH_SIZE = 64
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
```
* The values remain the same, only the name of the variables are changed to remove any confusion.
* We now load the dataset and create the fields, and the tabular dataset as done earlier.
```python
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
```
* Now we create our new iterators which will generate the new batches for training, validation, and testing.
```python
training_set_iterC = Iterator(train, batch_size=CLASSIFIER_TRAIN_BATCH_SIZE, device=device, train=True, shuffle=True, sort=False)
valid_set_iterC = Iterator(valid, batch_size=CLASSIFIER_VAL_BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)
test_set_iterC = Iterator(test, batch_size=CLASSIFIER_TEST_BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)
```
* The complete code for this section would look like:
<details>
 <summary>View Code</summary>

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
</details>


* Now that our hyperparameters are set, we create the replica model and load the weights. 
```python
CLASSIFIER_model = ROBERTA()
CLASSIFIER_model = CLASSIFIER_model.to(device)
preTrained = torch.load("./best_pre_train_model.pth")
CLASSIFIER_model.load_state_dict(preTrained, strict=False)
```
* The weights are loaded using ```torch.load()``` function, where the path to the weights is passed as the argument.
* Once are model is ready and weights are loaded, we can begin with training the final model.
* The ```classifier(model, optimizer, training_set_iter, valid_set_iter, scheduler, num_epochs)``` function is used to train the classifier.
* The major steps for this function are same as the one used for pre-training, but we'll go through them once again just to be sure.
* We first define our lists that store the training loss, validation loss and epochs.
```python
train_loss_list = []
val_loss_list = []
epc_list = []
```
* Once that is done, we freeze the weights of the RoBERTa layers again.
```python
for param in model.roberta.parameters():
        param.requires_grad = False
    
model.train()
```
* Also remember, that these steps take place inside the classifier function.
* We now define our loss function and set the best validation loss to infinity, just like we did in the pre-training step.
```python
criterion = torch.nn.CrossEntropyLoss()
best_valid_loss = float('Inf')
```
* Once these steps are set up, we begin our epoch loop:
```python
for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0
```
* As discussed earlier, the training and validation loss for each epoch is set to 0.0 at the beginning.
* We then use the model to make the predictions, and calculate the loss using the real values and the predictions.
```python
for (review, label), _ in training_set_iter:
            mask = (review != PAD_INDEX).type(torch.uint8)
            
            y_pred = model(input_ids=review, attention_mask=mask)
            
            loss = criterion(y_pred, label)
```
* Once our losses are calculated, we can update the weights, and schedule the learning rate and optimizer too,
```python
loss.backward()
# Optimizer and scheduler step
optimizer.step()    
scheduler.step()
optimizer.zero_grad()
# Update train loss and global step
train_loss += loss.item()
```
* Once the batches for the epoch are completed, we can use the model to test the validation set.
```python
model.eval()
with torch.no_grad():                    
 for (review, target), _ in valid_set_iter:
  mask = (review != PAD_INDEX).type(torch.uint8)
  y_pred = model(input_ids=review, attention_mask=mask)
  loss = criterion(y_pred, target)
  valid_loss += loss.item()
```
* We calculate the validation loss for the model, and now can proceed further to calculate the aggregate losses, print the training summary and saving the model if needed.
```python
train_loss = train_loss / len(training_set_iter)
valid_loss = valid_loss / len(valid_set_iter)
```
* We first calculate the aggregate losses which are used to plot the graphs as above.
```python
model.train()
train_loss_list.append(train_loss)
val_loss_list.append(valid_loss)
epc_list.append(epoch)
# print summary
print('Epoch [{}/{}], Pre-Training Loss: {:.4f}, Val Loss: {:.4f}'
.format(epoch+1, num_epochs, train_loss, valid_loss))
```
* We append the losses to our lists, and then proceed with printing the model summary.
```python
if best_valid_loss > valid_loss:
 best_valid_loss = valid_loss 
 # Saving Pre-Trained Model as .pth file
 torch.save({'model_state_dict': model.state_dict()}, "./final_model.pth")
```
* The code above saves the model if the validation loss for the epoch was lowest, in case the current validation score is lower, the model is saved.
* We conclude the training by setting the weights of RoBERTa layers to trainable again.
```python
for param in model.roberta.parameters():
 param.requires_grad = True
```
* The complete implementation looks something like:
<details><summary>View Code</summary>

```python
train_loss_list = []
val_loss_list = []
epc_list = []

def classifier(model, optimizer, training_set_iter, valid_set_iter, scheduler, num_epochs):
    
    # Pretrain linear layers, do not train bert
    for param in model.roberta.parameters():
        param.requires_grad = False
    
    model.train()
    
    # Initialize losses and loss histories
    
    criterion = torch.nn.CrossEntropyLoss()
    best_valid_loss = float('Inf')
    # Train loop
    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0 
        for (review, label), _ in training_set_iter:
            mask = (review != PAD_INDEX).type(torch.uint8)
            
            y_pred = model(input_ids=review, attention_mask=mask)
            
            loss = criterion(y_pred, label)
   
            loss.backward()
            
            # Optimizer and scheduler step
            optimizer.step()    
            scheduler.step()
                
            optimizer.zero_grad()
            
            # Update train loss and global step
            train_loss += loss.item()
                
        model.eval()
        
        with torch.no_grad():                    
            for (review, target), _ in valid_set_iter:
                mask = (review != PAD_INDEX).type(torch.uint8)
                
                y_pred = model(input_ids=review, attention_mask=mask)
                
                loss = criterion(y_pred, target)
                
                valid_loss += loss.item()

        # Store train and validation loss history
        train_loss = train_loss / len(training_set_iter)
        valid_loss = valid_loss / len(valid_set_iter)
        
        model.train()
        train_loss_list.append(train_loss)
        val_loss_list.append(valid_loss)
        epc_list.append(epoch)
        # print summary
        print('Epoch [{}/{}], Pre-Training Loss: {:.4f}, Val Loss: {:.4f}'
              .format(epoch+1, num_epochs, train_loss, valid_loss))
        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss 
            # Saving Pre-Trained Model as .pth file
            torch.save({'model_state_dict': model.state_dict()}, "./final_model.pth")
    
    # Set bert parameters back to trainable
    for param in model.roberta.parameters():
        param.requires_grad = True
     
    
        
    print('Training done!')
```

</details>


* Now that our training function is all set, we can train our model and implement Transfer Learning.
```python
CLASSIFIER_NUM_EPOCHS = 20
steps_per_epoch = len(training_set_iter)

CLASSIFIER_optimizer = AdamW(CLASSIFIER_model.parameters(), lr=1e-4)
CLASSIFIER_scheduler = get_linear_schedule_with_warmup(CLASSIFIER_optimizer, 
                                            num_warmup_steps=steps_per_epoch*1, 
                                            num_training_steps=steps_per_epoch*CLASSIFIER_NUM_EPOCHS)
print('Training starts')
classifier(model=CLASSIFIER_model,optimizer=CLASSIFIER_optimizer, training_set_iter=training_set_iter, valid_set_iter=valid_set_iter,  scheduler=CLASSIFIER_scheduler, num_epochs=CLASSIFIER_NUM_EPOCHS)
```

* The implementation of this section can be done as:
<details><summary>View Code</summary>

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
</details>


* The above code initiates the training process, and we get the final training summary as shown below. Notice that we have loaded our model in this section.
<p align="center">
 <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/FinalTrainStats.png" />
</p> 

* Just like in the previous section, we can use our lists to plot the graph of training and validation losses for the epochs, this can be done using: 
```python
plt.figure(figsize=(10, 8))
plt.plot(epc_list, train_loss_list, label='Train')
plt.plot(epc_list, val_loss_list, label='Valid')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=14)
plt.show()
```
* We get the following image as the plot for the trend of the losses:
<p align="center">
  <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/TrainValidLossGraphFinalTraining.png" />
</p>

* Now that we have implemented Transfer Learning, it is time to evaluate the model and check if there were any improvements to the previous scores.
* For that we use the evaluation function again as earlier, the code for the evaluation function is mentioned again for your help:
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
* The only thing left is to evaluate the final model, we do that using the following piece of code:
```python
evaluate(CLASSIFIER_model, test_set_iter)
```
* Remember to change the argument ```CLASSIFIER_model``` with the name of your own classifier, in case you follow our tutorial completely, there is no need for any change.
* The evaluation function return the following result:
<p align="center">
  <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/ConfusionMatrixFinal.png" />
</p>

* Also, our final model is saved, which we will use for deployment on the web application.

### Conclusions
 Upon comparing the results we can draw the following conclusions:
 * There is an improvement in f1-score for both the classes.
 * The accuracy in case of our final model is higher than the one trained before.
The above results prove that Transfer Learning shows an improvement in the results of training.


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


