# Introduction
Flask is a lightweight web framwork written in python. Flask is easy to get started for beginner. It is classified as a microframework because it does not require particular tools or libraries. It has no database abstraction layer, form validation, or any other components where pre-existing third-party libraries provide common functions.  
In this blog you will learn how to set up a Flask project and how to deploy Machine Learning model you have developed in previous blog. By the end of this blog you will be able to deploy any model using `Flask` on `Heroku`.

## Pre-Requisities
To implement the complete project you will need the following:
* Any operating system Linux, Windows, or Mac OS.
* Python 3+ installed https://www.python.org/
* Basic Python programming https://docs.python.org/3/tutorial/
* Basic Git knowledge https://git-scm.com/


## Step 1 - Instaling Flask and related package  
* Before installing Flask, we will create separate `python environment` for this project.
```python
python3 -m venv env
```
* To use this environment we need to activate it using folloying code.
```python
source venv/bin/activate
```
* Now environment is activated now we can download all packages.  
A few packages are requrired for this project like pytorch, numpy, pandas, transformers, and pickel.
```python
pip install flask pytorch torchvision numpy pandas transformer pickel
```

## Step 2 - File Structure  
We need to have a file structure for best practice.
```js
─── Flask
    ├── ml_model
    │    ├── modelFinal.pth ------- (Final saved model)
    │    ├── predict.py ----------- (Python script to create pipeline)
    │    └── vocab.pickel --------- (Contain English vocab used to create word sequence)
    └── app.py -------------------- (Main Flask File)
```

## Step 3 - Creating prediction pipeline.
We define our pipeline script in **`predict.py`** file under **`ml_model`** folder.  
Final pipeline is as follows - 
    **Data -> Pre-processing -> Model -> Prediction -> Final Result**  



<details> 
    <summary>Final code present here.</summary>
    <h3 style="display:inline-block"><summary>All Code to be written in <u><i>ml_model/predict.py</i></u> model </summary></h3>
    
```python
    import re
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    import torch.nn.functional as F
    from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup


    import os
    cwd = os.getcwd()

    import pickle
    b =  pickle.load(open(cwd + '\\ml_model\\vocab.pickle','rb'))

    vocab = b['vocab']

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
        return text

    class ROBERTA(torch.nn.Module):
        def __init__(self, dropout_rate=0.3):
            super(ROBERTA, self).__init__()
            self.roberta = RobertaModel.from_pretrained('roberta-base')
            self.d1 = torch.nn.Dropout(dropout_rate)
            self.l1 = torch.nn.Linear(768, 64)
            self.bn1 = torch.nn.LayerNorm(64)
            self.d2 = torch.nn.Dropout(dropout_rate)
            self.l2 = torch.nn.Linear(64, 2)

        def forward(self, input_ids, attention_mask):
            _, x = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            x = self.d1(x)
            x = self.l1(x)
            x = self.bn1(x)
            x = torch.nn.Tanh()(x)
            x = self.d2(x)
            x = self.l2(x)
            return x


    model = ROBERTA()

    state_dict = torch.load(cwd + '\\ml_model\\final_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)


    def pred(text):
        print("Text Received =>", text)
        text = text_preprocess(text)
        word_seq = np.array([vocab[word] for word in text.split() 
                          if word in vocab.keys()])
        word_seq = np.expand_dims(word_seq,axis=0)
        t = torch.from_numpy(word_seq).to(torch.int64)
        mask = (t != 1).type(torch.uint8)

        output = model(t, attention_mask=mask)
        print("Got output - ",output)
        pro = torch.argmax(output, axis=-1).tolist()[0]
        status = "positive" if pro == 1 else "negative"
        return status  
```
    
</details>


1. **Getting data.**
    * Data will be recieved in JSON formate(we will disscus later about how to recieve data).
    ```json 
    {
           "review": "Sample review"
     }
    ```
2. **Pre-Processing**  
    We pre-processed the text by lower casing, removing spectial character, etc. We will use the function given below.
    ```python
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
        return text
    ```
3. **Defining and Loading Machine Learning model**
    * For this problem we first define our Model archichetire which is based on `BoREBTa` and then loading pre-traing model we saved in previous blog.
    
4. **Finnaly we wrap all whole pipeline in a single Function given below.**  
    
    ```python
    def pred(text):
        text = text_preprocess(text)
        word_seq = np.array([vocab[word] for word in text.split() 
                          if word in vocab.keys()])
        word_seq = np.expand_dims(word_seq,axis=0)
        t = torch.from_numpy(word_seq).to(torch.int64)
        mask = (t != 1).type(torch.uint8)
        output = model(t, attention_mask=mask)
        print("Got output - ",output)
        pro = torch.argmax(output, axis=-1).tolist()[0]
        status = "positive" if pro == 1 else "negative"
        return status
    ```
    
## Step 4 - Final flask script.
This final script is written `app.py` file. This file is 
