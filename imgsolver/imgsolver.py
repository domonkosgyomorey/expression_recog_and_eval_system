import os
import matplotlib.pyplot as plt

import imgsolver.expr2seg_img as e2s
import keras
import numpy as np


class ImgSolver:

    def __init__(self):
        self.category_indices_v4 = None
        self.category_indices_v3 = None
        
        self.digit_indices = None
        self.operator_indices = None
        self.paren_indices = None
        self.trig_log_indices = None

        self.category_model_path_v4 = 'models/category_class_v4'
        self.category_model_path_v3 = 'models/category_class_v3'
        
        self.digit_model_path = 'models/digit_class_v4'
        self.operator_model_path = 'models/operator_class_v4'
        self.paren_model_path = 'models/paren_class_v4'
        self.trig_log_model_path = 'models/trig_log_class_v4'
        
        self.model_ext = '.model.keras'
        self.indices_ext = '_indices.txt'
        
        self.category_model_v4 = None
        self.category_model_v3 = None
        self.digit_model = None
        self.operator_model = None
        self.paren_model = None
        self.trig_log_model = None
        
        if os.path.exists(self.category_model_path_v4+self.model_ext) and os.path.exists(self.category_model_path_v3+self.model_ext) and os.path.exists(self.digit_model_path+self.model_ext) and os.path.exists(self.operator_model_path+self.model_ext) and os.path.exists(self.trig_log_model_path+self.model_ext):
            self.category_model_v4:keras.models.Sequential = keras.models.load_model(self.category_model_path_v4+self.model_ext)
            self.category_model_v3:keras.models.Sequential = keras.models.load_model(self.category_model_path_v3+self.model_ext)
            self.digit_model:keras.models.Sequential = keras.models.load_model(self.digit_model_path+self.model_ext)
            self.operator_model:keras.models.Sequential = keras.models.load_model(self.operator_model_path+self.model_ext)
            self.paren_model:keras.models.Sequential = keras.models.load_model(self.paren_model_path+self.model_ext)
            self.trig_log_model:keras.models.Sequential = keras.models.load_model(self.trig_log_model_path+self.model_ext)
        else:
            raise Exception("Some models are missing")

        if os.path.exists(self.category_model_path_v4+self.indices_ext) and os.path.exists(self.category_model_path_v3+self.indices_ext) and os.path.exists(self.digit_model_path+self.indices_ext) and os.path.exists(self.operator_model_path+self.indices_ext) and os.path.exists(self.paren_model_path+self.indices_ext) and os.path.exists(self.trig_log_model_path+self.indices_ext):    
            with open(self.category_model_path_v4+self.indices_ext, 'r') as file:
                content = file.read()
                self.category_indices_v4 = eval(content)
            
            with open(self.category_model_path_v3+self.indices_ext, 'r') as file:
                content = file.read()
                self.category_indices_v3 = eval(content)

            with open(self.digit_model_path+self.indices_ext, 'r') as file:
                content = file.read()
                self.digit_indices = eval(content)

            with open(self.operator_model_path+self.indices_ext, 'r') as file:
                content = file.read()
                self.operator_indices = eval(content)
            
            with open(self.paren_model_path+self.indices_ext, 'r') as file:
                content = file.read()
                self.paren_indices = eval(content)
            
            with open(self.trig_log_model_path+self.indices_ext, 'r') as file:
                content = file.read()
                self.trig_log_indices = eval(content)
        else:
            raise Exception('Missing indices files')

    def eval(self, img):
        seg_and_x = e2s.expr2segm_img(img)
        sorted_seg = map(lambda x: x[0], sorted(seg_and_x, key=lambda x: x[1]))
        for segment in sorted_seg:
            segment = segment.astype('float32')
            plt.imshow(segment)
            plt.show()
            segment = np.expand_dims(segment, axis=0)
            category_pred_v4 = self.category_model_v4.predict(segment) 
            category_pred_v3 = self.category_model_v3.predict(segment) 
            category_v4 = self.category_indices_v4[np.argmax(category_pred_v4)]           
            category_v3 = self.category_indices_v3[np.argmax(category_pred_v3)]           
            print('Category distribution v4: ', category_pred_v4)
            print('Category distribution v3: ', category_pred_v3)
            print('Category prediction v4: ', category_v4, '\n')
            print('Category prediction v3: ', category_v3, '\n')
            category_pred = category_pred_v3*category_pred_v4
            category = self.category_indices_v4[np.argmax(category_pred)]
            
            prediction = None
            pred_dist = None
            if category=='digit':
                pred_dist = self.digit_model.predict(segment)
                prediction = self.digit_indices[np.argmax(pred_dist)]
            elif category=='operator':
                pred_dist = self.operator_model.predict(segment)
                prediction = self.operator_indices[np.argmax(self.operator_model.predict(segment))]
                if prediction == ',':
                    pred_dist = [0.5]
                    prediction = '1'
            elif category=='paren':
                pred_dist = self.paren_model.predict(segment)
                prediction = self.paren_indices[np.argmax(self.paren_model.predict(segment))]
            elif category=='trig_log':
                pred_dist = self.trig_log_model.predict(segment)
                prediction = self.trig_log_indices[np.argmax(self.trig_log_model.predict(segment))]
            
            print('Prediction distribution: ', pred_dist)
            print('Prediction: ', prediction)
            print('Confidence: ', np.max(pred_dist)*np.max(category_pred))

