# Programmer: Peyton Urquhart
# Project: Image Emotion Classifier
# Date: 3/31/2022

#   image_processor.py
#   
#   For each unique image in the samples/raw directory, identify all faces
#   in the image, process the images, and write new images to the corresponding
#   samples/processed directory.
#
#   The amount of output images could be more than the amount of input images.
#
#   To process all images in samples/raw: 'python3 image_processor.py'
#   To process a certain type of images: 'python3 image_processor.py <emotion>'
#       For example: 'python3 image_processor.py anger'

import definitions as DF
from face_finder import FaceFinder
from os import listdir
import sys

RAW_PATH = '../samples/raw/'
PROCESSED_PATH = '../samples/processed/'
OUTPUT_FILETYPE = '.jpg'

# Get paths to directories for each type of emotion
# returns a list of (emotion, emotion source path, emotion output path)
def get_sample_paths():
    paths = []
    for emotion in DF.EMOTIONS:
        paths.append((emotion, RAW_PATH+emotion, PROCESSED_PATH+emotion))
    return paths

# Given an image path, returns true if the image is acceptable for processing
def is_acceptable_image(stem):
    try:
        e1 = stem[-5:]
        e2 = stem[-4:]
    except:
        return False
    if e1 == '.jpeg' or e2 == '.jpg':
        return True
    return False

# Returns true if the script was run with arguments
def with_args():
    if len(sys.argv) < 2:
        return False
    if sys.argv[1] not in DF.EMOTIONS:
        raise Exception(f'Bad argument: {sys.argv[1]}')
    return True

if not with_args():
    sample_paths = get_sample_paths()
else:
    sample_paths = [(sys.argv[1], RAW_PATH+sys.argv[1], PROCESSED_PATH+sys.argv[1])]

faceFinder = FaceFinder(log=True)

# for each emotion type
for (emotion, raw_dir, processed_dir) in sample_paths:
    contents = listdir(raw_dir)
    # for each file in a certain emotion
    num_images = 0
    for image_file in contents:
        if not is_acceptable_image(image_file):
            continue
        abs_path = raw_dir+'/'+image_file
        print(f'[{emotion}]: {abs_path}')
        faces = faceFinder.get_faces_from_file(abs_path)
        # for each face detected in a file
        for i in range(len(faces)):
            output_path = processed_dir+'/'+emotion+"_"+str(num_images)+"_"+str(i)+OUTPUT_FILETYPE
            print(f'    [output {i}]: {output_path}')
            faces[i].save(output_path)
        num_images += 1