import definitions as DF
from face_finder import FaceFinder
from create_hog import make_hog_from_processed_data
import numpy as np

class PredictImage:
    def __init__(self, clf):
        self.clf = clf
        self.finder = FaceFinder(log=False)

    def process_predict(self, image, fromFile, use_probability=True):
        if fromFile == True:
            hogs = self.process_from_file(image)
        else:
            hogs = self.process_from_data(image)
        if use_probability:
            return self.predict_proba(hogs)
        return self.predict(hogs)

    def process_from_data(self, data):
        hogs = []
        faces = self.finder.get_faces_from_ndarray(np.array(data))
        for face in faces:
            data = np.array(face)
            hog = make_hog_from_processed_data(data, False)
            hogs.append(np.array(hog))
        return hogs
    
    def process_from_file(self, image_path):
        hogs = []
        faces = self.finder.get_faces_from_file(image_path)
        for face in faces:
            data = np.array(face)
            hog = make_hog_from_processed_data(data, False)
            hogs.append(np.array(hog))
        return hogs

    def predict(self, hogs):
        predictions = []
        for h in hogs:
            pred = self.clf.predict([h])
            predictions.append(pred[0])
        return predictions

    def predict_proba(self, hogs):
        predictions = []
        for h in hogs:
            pred = self.clf.predict_proba([h])
            predictions.append(pred[0])
        return predictions

    def format_probabilities(self, pred):
        d = {}
        for emotion in DF.EMOTIONS:
            d[emotion] = 0.0
        fpred = self._truncate_probabilities(pred)
        for i in range(len(fpred)):
            ce = DF.emotion_from_int(i)
            d[ce] = fpred[i]*100
        return sorted([(k,v) for k,v in d.items()], key=lambda x: x[1], reverse=True)

    def _truncate_probabilities(self, arr, amount=3):
        r = []
        for i in range(len(arr)):
            r.append(round(arr[i], amount))
        return r
        
