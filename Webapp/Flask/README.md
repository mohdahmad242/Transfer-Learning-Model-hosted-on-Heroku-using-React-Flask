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
