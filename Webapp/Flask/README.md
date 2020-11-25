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
    <h3 style="display:inline-block"><summary>All Code to be written in <u><i>ml_model/predict.py</i></u> fie. </summary></h3>
    
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
This final script is to be written in `app.py` file. This file will handel all HTTP requests we are going to use.    
In this file we will `import pred()` function we created in pipeline section then `import Flask` to create an app instance.
```python
    app = Flask(__name__)
```
We will use this instance to handel HTTP request. First we will define a route decorator `@app.route()` then pass API end name, here we use **`/predict`**.
Since, we will be receiving our text data on this rout, here will we pass method as `POST`.  
Learn about basics of newtorking if you don't know [Learn Here](https://www.toolsqa.com/rest-assured/rest-routes/)
```python
    @app.route('/predict', methods=['POST'])
```
Now, we will define a fuction which will be executed when someone make POST request on `/predict` route. In this function we will take our POST route data and pass to the `pred()` function we imported from **`ml_model/predict.py`** file.
```python
    def home():
        review = request.json['review']
        prediction = pred(review)
        print(prediction)
        return prediction
```
<details> 
    <summary>Final code for <b>app.py</b> file is present here.</summary>
    <h3 style="display:inline-block"><summary>All this code to be written in <u><i>app.py</i></u> fie. </summary></h3>
    
```python
    from ml_model.predict import pred

    import os
    from flask import Flask, render_template, request, make_response, jsonify, send_file

    app = Flask(__name__)

    # Set up the main route
    @app.route('/predict', methods=['POST'])
    def home():
        print("Action Initiated")
        review = request.json['review']
        prediction = pred(review)
        print(prediction)
        return prediction

    if __name__ == '__main__':
        app.run()

```
    
</details>

## Step 4 - Testing final app.
To run the Flask open terminal in the project directory and type following command in terminal.
```bash
    flask run
```
Now, to test your final API you can use any API Client (Postman, Insomnia, etc).  
We are using `Postman`, you can learn about it here - https://www.postman.com/

# Diployment on Heroku.
We expect you have GitHub account and you know how to create repository. If not, [learn here](https://guides.github.com/activities/hello-world/)
* Create a file naming **`Procfile`** in main directory. 
    This file specify which command to run at app startup, for our app write this command
    ```bash
    web: gunicorn --bind 0.0.0.0:$PORT main_app:app
    ```
> **IMPOERTANT** In your requrement.txt file remove torch and put these and same for touchvision   
    > https://download.pytorch.org/whl/cpu/torch-1.6.0%2Bcpu-cp37-cp37m-linux_x86_64.whl  
    > https://download.pytorch.org/whl/cpu/torchvision-0.7.0%2Bcpu-cp37-cp37m-linux_x86_64.whl
    <details> 
    <summary>Explanation with Screenshot</summary>
    `requirements.txt` file is used by the heroku server to download all packages. If we specify `torch == 1.6.0`, then it will download whole pytorch library. 
    Since we are using free version of heroku, we have only `CPU` support not `GPU`. Pytorch library comes with all files required for GPU support, so we need to download only 
    `CPU` specific files. As Heroku only provide `500 Mb` storage in free version and complete Pytorch library is more than 600 Mb, that why we need only CPU version which far     less space.
    </details>
* Now create a repository and push all code on github repo.
* Create Heroku account https://signup.heroku.com/
* Link you github account with heroku.
* Choose the github repository where you push all your code.
* Deploy. 
    
    

