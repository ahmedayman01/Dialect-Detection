import os
import json
from datetime import datetime
from threading import Thread, current_thread, get_ident
from typing import Dict, List, Union
import joblib
from sklearn.model_selection import train_test_split
import re
import numpy as np
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras.layers import LSTM, SpatialDropout1D
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

__storage_path = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'storage')

__model_path = os.path.join(__storage_path, 'checkpoints')
# print(os.path.join(__model_path, 'my_checkpoint'))
__encoder_path = os.path.join(__storage_path, 'classes.npy')

def _txt_prep(txt: List[str]):
    voc_size = 80000
    sent_length = 250

    corpus = []
    for i in txt:
        i = re.sub("\n", "", i)
        corpus.append(i)

    onehot_repr = [one_hot(words, voc_size) for words in corpus]
    embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
    return embedded_docs

def _create_model():
    voc_size = 80000
    sent_length = 250
    embedding_vector_features = 100
    model = Sequential()
    model.add(Embedding(voc_size + 1, embedding_vector_features, input_length=sent_length))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(18, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if os.path.exists(__encoder_path):
    encoder = LabelEncoder()
    encoder.classes_ = np.load(__encoder_path, allow_pickle=True)

# else:
#     encoder = LabelEncoder()

# model = _create_model()
# model.load_weights(os.path.join(__model_path, 'my_checkpoint'))
#
# txt = ["لكن بالنهاية ينتفض يغير","مبين من كلامه خليجي"]
# embedded_txt = _txt_prep(txt)
# texts = np.array(embedded_txt)
# response = []
# if model:
#     pred = model.predict(texts)
#     predict_classes = np.argmax(pred, axis=1)
#     print(predict_classes)
#     pred = encoder.inverse_transform(predict_classes)
#     print(pred)
#     for i in range(len(pred)):
#         row_pred = {'text': txt[i], 'predictions': pred[i]}
#         response.append(row_pred)
#
# print(response)
print(np.load(__encoder_path, allow_pickle=True))

print(encoder.transform(['EG',"Kw"]))