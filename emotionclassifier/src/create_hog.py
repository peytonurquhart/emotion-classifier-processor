# Programmer: Peyton Urquhart
# Project: Image Emotion Classifier
# Date: 3/31/2022

#   create_hog.py
#
#   This file is used to create hog data from either a processed image
#   at a certain path, or from processed image data in array format

from skimage.feature import hog
from skimage.io import imread
from matplotlib import pyplot as plt

def make_hog_from_path(path, vizualize=False):
    image_data = imread(path, as_gray=True)
    if vizualize:
        image_hog, image_hog_viz = hog(image_data, visualize=vizualize)
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(8,6)
        for a in ax:
            a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax[0].imshow(image_data, cmap='gray')
        ax[1].imshow(image_hog_viz, cmap='gray')
        print(image_hog)
        plt.show()
        return image_hog
    return hog(image_data, visualize=False)

def make_hog_from_processed_data(image_data, vizualize=False):
    if vizualize:
        image_hog, image_hog_viz = hog(image_data, visualize=vizualize)
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(8,6)
        for a in ax:
            a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax[0].imshow(image_data, cmap='gray')
        ax[1].imshow(image_hog_viz, cmap='gray')
        print(image_hog)
        plt.show()
        return image_hog
    return hog(image_data, visualize=False)