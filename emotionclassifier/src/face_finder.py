# Programmer: Peyton Urquhart
# Project: Image Emotion Classifier
# Date: 3/31/2022

#   face_finder.py
#   
#   Contains the FaceFinder class, which given an image, returns a list of images
#   each containing a separate face found in the origional. 
#   The images will be rescaled to match 'output_dimensions' and can optionally be
#   converted to grayscale.

from PIL import Image
import face_recognition as FR

class FaceFinder:
    def __init__(self, log, output_dimensions = (100, 100), grayscale = True):
        self.log = log
        self.output_dim = output_dimensions
        self.grayscale = grayscale

    def get_faces_from_file(self, file):
        image = FR.load_image_file(file)
        locations = FR.face_locations(image)
        self.logMessage(str(f'detected {len(locations)} faces.'))
        faces = []
        for cl in locations:
            t, r, b, l = cl
            fi = image[t:b, l:r]
            renderable = Image.fromarray(fi)
            renderable = renderable.resize(self.output_dim)
            if self.grayscale:
                renderable = renderable.convert('L')
            faces.append(renderable)
        return faces

    def get_faces_from_ndarray(self, image):
        locations = FR.face_locations(image)
        self.logMessage(str(f'detected {len(locations)} faces.'))
        faces = []
        for cl in locations:
            t, r, b, l = cl
            fi = image[t:b, l:r]
            renderable = Image.fromarray(fi)
            renderable = renderable.resize(self.output_dim)
            if self.grayscale:
                renderable = renderable.convert('L')
            faces.append(renderable)
        return faces

    def logMessage(self, message):
        if self.log:
            print(message)

