from flask import Flask, render_template, request
# from sklearn.externals import joblib
import joblib


import pandas as pd
import numpy as np
app=Flask(__name__)
'''
@app.route('/test')


def test():
    return "Flask is being used for development"
'''

#load model_prediction
#We can load model here such that it will not be loaded again if the page refreshes, iproves the performance
from tensorflow.keras.models import load_model
import string
import pandas as pd
import numpy as np

model = load_model('30_real_user_weight.h5')
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

def create_vocab_set():
    alphabet = (list(string.ascii_lowercase) + list(string.digits) +
                list(string.punctuation) + ['\n'] + [' '] )
    vocab_size = len(alphabet)
    check = set(alphabet)

    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, check

maxlen = 140
vocab, reverse_vocab, vocab_size, check =create_vocab_set()

def encode_data(x, maxlen, vocab, vocab_size, check):

    input_data = np.zeros((len(x), maxlen, vocab_size))
    for dix, sent in enumerate(x):
        counter = 0
        sent_array = np.zeros((maxlen, vocab_size))
        chars = list(sent.lower())
        for c in chars:
            if counter >= maxlen:
                pass
            else:
                char_array = np.zeros(vocab_size, dtype=np.int)
                if c in check:
                    ix = vocab[c]
                    char_array[ix] = 1
                sent_array[counter, :] = char_array
                counter += 1
        input_data[dix, :, :] = sent_array

    return input_data


@app.route('/')


def home():
    return render_template('home.html')

@app.route("/predict",methods=['GET','POST'])


def predict():
    if request.method == 'POST':
        try:
            text=(request.form['text'])
            type_here=[]
            type_here.append(text)
            typr_here=pd.DataFrame(type_here)
            typr_here = encode_data(type_here, maxlen, vocab, vocab_size, check)
            y_pred = model.predict(typr_here)
            y_pred=pd.DataFrame(y_pred)
            y_pred=y_pred.eq(y_pred.where(y_pred != 0).max(1), axis=0).astype(int)
            y_pred=y_pred.iloc[:,:].values
            result=[]
            for i in range(0,len(y_pred)):
                for j in range(0,len(y_pred[0])):
                    if(y_pred[i][j]==1):
                        result.append(j)
            author=encoder.inverse_transform(result)
        except valueError:
            return "please Check if the values are entered correctly"
    return render_template('predict.html', prediction=author[0])

if __name__=="__main__":
    app.run()
