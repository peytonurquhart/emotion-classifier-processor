# Programmer: Peyton Urquhart
# Project: Image Emotion Classifier
# Date: 3/31/2022

#   create_model.py
#
#   The entry point for model training. Before you get here, you must process raw image data
#   to do this, see image_processor.py.
#
#   You may need to generate the X and y data with gen_hog_data() and the use export_to_csv() to
#   create 'X.csv' and 'y.csv' in the 'samples' directory. Once these files have been genereated,
#   load_from_csv() can load the X and y in directly.
#
#   'python3 create_model.py'

from collections import Counter
from load_processed_data import gen_hog_data, export_to_csv, load_from_csv
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import joblib
import sys

MODEL_PATH = '../models'

def get_best_hyperparamaters(X, y):
    clf = MLPClassifier(max_iter=500)
    parameter_space = {
        'solver': ['adam'],
        'activation': ['logistic'],
        'hidden_layer_sizes': [(32,), (100,)],
        'learning_rate': ['constant'],
        'alpha': [0.0001]
        }
    clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)
    clf.fit(X, y)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print('Best parameters found:', clf.best_params_)
    
    return clf.best_estimator_

def save_model(name, clf):
    joblib.dump(clf, MODEL_PATH+'/'+name, compress=9)
    print(f'model saved successfully: {MODEL_PATH}/{name}')

def train_model(X, y):
    clf = get_best_hyperparamaters(X, y)
    print('model has been fit')
    return clf

def oversample(X, y):
    oversample = RandomOverSampler(random_state=0)
    X, y = oversample.fit_resample(X, y)
    return X, y

def create_model(name, generate_new_data=False):
    if generate_new_data:
        print('loading data from processed images')
        X, y = gen_hog_data()
        export_to_csv(X, y)
    else:
        print('loading data from .csv')
        X, y = load_from_csv()
    print('data loaded successfully')
    X, y = oversample(X, y)
    clf = train_model(X, y)
    save_model(name, clf)

if len(sys.argv) < 2:
    raise Exception('you must name the new model')
gen_new = False
if len(sys.argv) > 2 and sys.argv[2] == '-true':
    gen_new = True
model_name = sys.argv[1]
create_model(model_name, gen_new)