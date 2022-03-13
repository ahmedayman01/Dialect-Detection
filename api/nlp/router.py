from __future__ import annotations
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nlp.ml import MlTrainer
from nlp.dl import DlTrainer

app = FastAPI()
ml_trainer = MlTrainer()
dl_trainer = DlTrainer()


class TrainingData(BaseModel):
    texts: List[str]
    labels: List[Union[str, int]]


class TestingData(BaseModel):
    texts: List[str]


class QueryText(BaseModel):
    text: str


class StatusObject(BaseModel):
    status: str
    timestamp: str
    classes: List[str]
    evaluation: Dict


class PredictionObject(BaseModel):
    text: str
    predictions: str


class PredictionsObject(BaseModel):
    predictions: List[PredictionObject]


@app.get("/ml-status", summary="Get current status of the system")
def get_ml_status():
    status = ml_trainer.get_status()
    return StatusObject(**status)


@app.get("/dl-status", summary="Get current status of the system")
def get_dl_status():
    status = DlTrainer.get_status()
    return StatusObject(**status)


@app.post("/ml-train", summary="Train a new ML model")
def ml_train(training_data: TrainingData):
    try:
        ml_trainer.train(training_data.texts, training_data.labels)
        status = ml_trainer.get_status()
        return StatusObject(**status)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/dl-train", summary="Train a new DL model")
def dl_train(training_data: TrainingData):
    try:
        DlTrainer.train(training_data.texts, training_data.labels)
        status = DlTrainer.get_status()
        return StatusObject(**status)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ml-predict", summary="Predict single input with ML Model")
def ml_predict(query_text: QueryText):
    try:
        prediction = ml_trainer.predict([query_text.text])[0]
        return PredictionObject(**prediction)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ml-predict-batch", summary="predict a batch of sentences with ML Model")
def ml_predict_batch(testing_data: TestingData):
    try:
        predictions = ml_trainer.predict(testing_data.texts)
        return PredictionsObject(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/dl-predict", summary="Predict single input with DL Model")
def dl_predict(query_text: QueryText):
    try:
        prediction = dl_trainer.predict([query_text.text])[0]
        return PredictionObject(**prediction)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/dl-predict-batch", summary="predict a batch of sentences with DL Model")
def dl_predict_batch(testing_data: TestingData):
    try:
        predictions = dl_trainer.predict(testing_data.texts)
        return PredictionsObject(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/")
def home():
    return ({"message": "System is up"})
