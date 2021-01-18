## This code aims to detect emotions from text using LSTM

!pip install tweet-preprocessor
!pip install nltk
!pip install scikit-learn
!pip install numpy
!pip install tensorflow
!pip install 3to2

## Import required packages

import pandas as pd
import preprocessor.api as p
import re
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn import metrics

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU,SimpleRNN
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
print(tf.__version__)

df = pd.read_csv("text_emotion.csv")
df.head()

## Step 1: Data pre-processing


## Remove mentions and "#" symbol in tweet
df['clean_content'] = df.content.apply(lambda x: re.sub('@(\w+)', '', x))
df['clean_content'] = df.clean_content.apply(lambda x: re.sub('#', "", x))

## Clean using the tweet-processing package, removing emojis and urls
df['clean_content'] = df.clean_content.apply(lambda x: p.clean(x))


## Remove unnecessary punctuation in the data, but tag ! and ?

def punctuation(val):
    punctuations = '''()-[]{};:'"\,<>./@#$%^&_~'''

    for x in val.lower():
        if x in punctuations:
            val = val.replace(x, " ")
        elif x == "!":
            val = val.replace(x, " XXEXLMARK ")
        elif x == "?":
            val = val.replace(x, " XXQUESMARK ")
    return val


df['clean_content'] = df.clean_content.apply(lambda x: punctuation(x))

## Remove empty data

df = df[df.clean_content != ""]

## Step 2: Modelling the data
## We will use a LSTM model and train it on this dataset

## First, we encode the emotion as numbers
sent_id = {"anger": 0, "hate": 1, "worry": 2, "sadness": 3, "neutral": 4, "empty": 5, "boredom": 6,
           "relief": 7, "happiness": 8, "love": 9, "enthusiasm": 10, "surprise": 11, "fun": 12}

df["sentiment_id"] = df['sentiment'].map(sent_id)

# Encode labels in column 'sent_id'.
label_encoder = preprocessing.LabelEncoder()
integer_encoded = label_encoder.fit_transform(df.sentiment_id)

onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y = onehot_encoder.fit_transform(integer_encoded)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(df.clean_content,Y, random_state=69, test_size=0.2, shuffle=True)

## Train the LSTM model


# Use the tokenizer that comes with Keras.
tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(list(X_train) + list(X_test))

max_len = 160
Epoch = 5

# Next, convert the text into padded sequences
X_train_pad = sequence.pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
X_test_pad = sequence.pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

w_idx = tokenizer.word_index

## Setup the model

embed_dim = 160
lstm_out = 250

model = Sequential()
model.add(Embedding(len(w_idx) +1 , embed_dim,input_length = X_test_pad.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(13, activation='softmax'))
#adam rmsprop
model.compile(loss = "categorical_crossentropy", optimizer='adam',metrics = ['accuracy'])
print(model.summary())

## Fit the LSTM Model

batch_size = 32
model.fit(X_train_pad, y_train, epochs = Epoch, batch_size=batch_size,validation_data=(X_test_pad, y_test))

def clean_text(val):
    val = p.clean(val)
    val = re.sub('@(\w+)','',val)
    val = re.sub('#',"", val)
    val = punctuation(val)
    return val


def get_sentiment(model,text):
    text = clean_text(text)
    #tokenize
    twt = tokenizer.texts_to_sequences([text])
    twt = sequence.pad_sequences(twt, maxlen=max_len, dtype='int32')
    sentiment = model.predict(twt,batch_size=1,verbose = 2)
    sent = np.round(np.dot(sentiment,100).tolist(),0)[0]
    result = pd.DataFrame([sent_id.keys(),sent]).T
    result.columns = ["sentiment","percentage"]
    result=result[result.percentage !=0]
    return result.sort_values(by = ['percentage'], ascending = False).sentiment.iloc[0]

y_test_pred = model.predict(X_test_pad)

## Calculate the AUC score (% of correct predictions)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

auc_lstm = roc_auc_score(y_test, y_test_pred)
print(auc_lstm)