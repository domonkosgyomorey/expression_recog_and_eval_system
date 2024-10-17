import os
import matplotlib.pyplot as plt

import imgsolver.expr_segmentation.expr2seg_img as e2s
import keras
import numpy as np


class ImgSolver:

    def __init__(self):
        return
        self.indices = None
        if os.path.exists('number_symbol_class.model.keras'):
            self.number_symbol_classifier = keras.models.load_model('number_symbol_class.model.keras')
        else:
            print('\"number symbol binary classifier\" model does not exists')
            
        if os.path.exists('number_class.model.keras'):
            self.number_classifier = keras.models.load_model('number_class.model.keras')
        else:
            print('\"number classifier\" model does not exists')
            
        if os.path.exists('symbol_class.model.keras'):
            self.symbol_classifier = keras.models.load_model('symbol_class.model.keras')
        else:
            print('\"symbol classifier\" model does not exists')
            
        with open('number_symbol_classifier_indices.txt', 'r') as file:
            content = file.read()
            self.num_sym_indices = eval(content)
            
        with open('number_class_indices.txt', 'r') as file:
            content = file.read()
            self.num_indices = eval(content)
        
        with open('symbol_class_indices.txt', 'r') as file:
            content = file.read()
            self.sym_indices = eval(content)
        

    def eval(self, img):
        seg_and_x = e2s.expr2segm_img(img)
        model : keras.models.Sequential = keras.models.load_model('number_classification_v3.model.keras')

        sorted_seg = map(lambda x: x[0], sorted(seg_and_x, key=lambda x: x[1]))
        for segment in sorted_seg:
            segment = segment.astype('float32')
            #plt.imshow(segment)
            #plt.show()
            segment = np.expand_dims(segment, axis=0)
            predicted_name = model.predict(segment)
            """
            class_prediction = self.number_symbol_classifier.predict(segment)
            predicted_class_name = self.num_sym_indices[np.argmax(class_prediction, axis=-1)[0]]
            print("Predicted class: ", predicted_class_name)
            
            if predicted_class_name == 'digit':
                prediction = self.number_classifier.predict(segment)
                predicted_name = self.num_indices[np.argmax(prediction, axis=-1)[0]]
            elif predicted_class_name == 'operator':
                prediction = self.symbol_classifier.predict(segment)
                predicted_name = self.sym_indices[np.argmax(prediction, axis=-1)[0]]
"""
            print("Predicted: ", predicted_name)
