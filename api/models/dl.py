import json
import os
import re
from datetime import datetime
from threading import Thread, get_ident
from typing import Dict, List

import numpy as np
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DlTrainer:

    def __init__(self) -> None:
        self.__voc_size = 10000
        self.__sent_length = 50
        self.__embedding_vector_features = 50
        self.__storage_path = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), os.path.join('storage','dl'))
        if not os.path.exists(self.__storage_path):
            os.mkdir(self.__storage_path)
        self.__status_path = os.path.join(
            self.__storage_path, 'dl_model_status.json')
        self.__model_path = os.path.join(
            self.__storage_path, 'checkpoints')
        self.__encoder_path = os.path.join(
            self.__storage_path, 'dl_classes.npy')

        if os.path.exists(self.__status_path):
            with open(self.__status_path) as file:
                self.model_status = json.load(file)

        else:
            self.model_status = {"status": "No Model found",
                                 "timestamp": datetime.now().isoformat(), "classes": [], "evaluation": {}}

        if os.path.exists(self.__model_path):
            model = self._create_model()
            model.load_weights(os.path.join(self.__model_path, 'my_checkpoint'))
            self.model = model

        else:
            self.model = None

        self._running_threads = []
        self._pipeline = None

        if os.path.exists(self.__encoder_path):
            self.encoder = LabelEncoder()
            self.encoder.classes_ = np.load(self.__encoder_path, allow_pickle=True)

        else:
            self.encoder = LabelEncoder()

    def _train_job(self, x_train: List[str], x_test: List[str], y_train: List[int],
                   y_test: List[int]):
        embedded_train = self._txt_prep(x_train)
        embedded_test = self._txt_prep(x_test)
        x_train, y_train = np.array(embedded_train), np.array(y_train)
        x_test, y_test = np.array(embedded_test), np.array(y_test)
        pred = self._pipeline.predict(x_test)
        predict_classes = np.argmax(pred, axis=1)
        pred = self.encoder.inverse_transform(predict_classes)
        self._pipeline.fit(x_train, y_train)
        report = classification_report(
            self.encoder.inverse_transform(y_test),pred , output_dict=True)
        classes = self._pipeline.classes_.tolist()
        self._update_status("Model Ready", classes, report)
        self._pipeline.save_weights(self.__model_path)
        self.model = self._pipeline
        self._pipeline = None
        thread_id = get_ident()
        for i, t in enumerate(self._running_threads):
            if t.ident == thread_id:
                self._running_threads.pop(i)
                break

    def train(self, texts: List[str], labels: List[str]) -> None:
        if len(self._running_threads):
            raise Exception("A training process is already running.")

        try:
            labels = [i.upper() for i in labels]
            labels = self.encoder.transform(labels)
        except:
            raise Exception("labels contains previously unseen labels")

        x_train, x_test, y_train, y_test = train_test_split(texts, labels)
        self._pipeline = self._create_model()

        # update model status
        self.model = None
        self._update_status("Training")

        t = Thread(target=self._train_job, args=(
            x_train, x_test, y_train, y_test))
        self._running_threads.append(t)
        t.start()

    def predict(self, texts: List[str]) -> List[Dict]:
        embedded_txt = self._txt_prep(texts)
        txt = np.array(embedded_txt)
        response = []
        if self.model:
            pred = self.model.predict(txt)
            predict_classes = np.argmax(pred, axis=1)
            pred = self.encoder.inverse_transform(predict_classes)
            for i in range(len(pred)):
                row_pred = {'text': texts[i], 'predictions': pred[i]}
                response.append(row_pred)
        else:
            raise Exception("No Trained model was found.")
        return response

    def get_status(self) -> Dict:
        return self.model_status

    def _update_status(self, status: str, classes: List[str] = [], evaluation: Dict = {}) -> None:
        self.model_status['status'] = status
        self.model_status['timestamp'] = datetime.now().isoformat()
        self.model_status['classes'] = classes
        self.model_status['evaluation'] = evaluation

        with open(self.__status_path, 'w+') as file:
            json.dump(self.model_status, file, indent=2)

    def _txt_prep(self, txt: List[str]):
        corpus = []
        for i in txt:
            i = re.sub("\n", "", i)
            corpus.append(i)

        onehot_repr = [one_hot(words, self.__voc_size) for words in corpus]
        embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=self.__sent_length)
        return embedded_docs

    def _create_model(self):
        model = Sequential()
        model.add(Embedding(self.__voc_size + 1, self.__embedding_vector_features, input_length=self.__sent_length))
        model.add(SpatialDropout1D(0.2))
        model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
        model.add(Dense(18, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
