# Programmer: Peyton Urquhart
# Project: Image Emotion Classifier
# Date: 3/31/2022

#   load_processed_data.py
#
#   Utility file for generating hog data from processed image files,
#   exporting image hog data to csv, and loading it back

import definitions as DF
from create_hog import make_hog_from_path
from os import listdir
import numpy as np

PROCESSED_PATH = '../samples/processed/'
X_CSV = '../samples/X.csv'
Y_CSV = '../samples/y.csv'

def get_sample_paths():
    paths = []
    for emotion in DF.EMOTIONS:
        paths.append((emotion, PROCESSED_PATH+emotion))
    return paths

def gen_hog_data():
    sample_paths = get_sample_paths()
    X = []
    y = []
    for (emotion, processed_dir) in sample_paths:
        contents = listdir(processed_dir)
        for image_file in contents:
            abs_path = processed_dir+'/'+image_file
            X.append(make_hog_from_path(abs_path, False))
            y.append(DF.emotion_to_int(emotion))
    return np.array(X), np.array(y)

def export_to_csv(X, y):
    np.savetxt(X_CSV, X, delimiter=',')
    np.savetxt(Y_CSV, y, delimiter=',')

def load_from_csv():
    X = np.loadtxt(X_CSV, delimiter=',')
    y = np.loadtxt(Y_CSV, delimiter=',')
    return X, y
