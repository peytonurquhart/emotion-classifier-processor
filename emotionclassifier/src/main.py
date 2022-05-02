# Programmer: Peyton Urquhart
# Project: Image Emotion Classifier
# Date: 3/31/2022

#   main.py
#
#   1. Populate samples/raw/* with images that correspond to the emotion required
#
#   2. Run 'python3 image_processor.py' to process each image. They will be printed to the samples/processed directory
#       optionally, run 'python3 image_processor.py <emotion_name>' to process each separately.
#
#   3. Run 'python3 create_model.py <model_name> -true' to create a new model from image data
#                                   OR
#      Run 'python3 create_model.py <model_name>' to create a new model from csv data
#
#   4. Run 'main.py <model_name> <image_path>
#

import sys
import face_recognition as FR
from emotion_classifier import EmotionClassifier

def main():
    if len(sys.argv) < 3:
        raise Exception('you must specify a model name and image file to use')

    model_name = sys.argv[1]
    path = sys.argv[2]

    app = EmotionClassifier(model_name)

    print("testing classification from raw image data...")
    # from raw image data ---
    data = FR.load_image_file(path)
    pred1 = app.predict_from_data(data)
    i = 0
    for face in pred1:
        print(f'face {i+1}:')
        for emotion in face:
            print(f'    {emotion}')
        i+=1
    # ---

    print("testing classification from saved image file...")
    # from file path ---
    pred2 = app.predict_from_path(path)
    i = 0
    for face in pred2:
        print(f'face {i+1}:')
        for emotion in face:
            print(f'    {emotion}')
        i+=1
    # ---


if __name__ == '__main__':
    main()