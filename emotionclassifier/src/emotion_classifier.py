# Programmer: Peyton Urquhart
# Project: Image Emotion Classifier
# Date: 3/31/2022

#   emotion_classifier.py
#
#   This file can be used as a factory to create an EmotionClassifier app
#   that could be used in other programs

import joblib
from os import listdir
from predict_image import PredictImage

MODELS_PATH = '../models'

class EmotionClassifier:
    def __init__(self, model_name):
        self.clf = self._load_model(model_name)
        self.predictor = PredictImage(self.clf)

    def predict_from_path(self, path):
        face_predictions = []
        preds = self.predictor.process_predict(path, True)
        for face in preds:
            face_predictions.append(self.predictor.format_probabilities(face))
        return face_predictions

    def predict_from_data(self, data):
        face_predictions = []
        preds = self.predictor.process_predict(data, False)
        for face in preds:
            face_predictions.append(self.predictor.format_probabilities(face))
        return face_predictions

    def _load_model(self, name):
        for n in listdir(MODELS_PATH):
            if n == name:
                return joblib.load(MODELS_PATH+'/'+name)
        raise Exception(f'could not find a model named: {name}')
