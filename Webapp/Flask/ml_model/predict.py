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
model.eval()

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

