import os
import matplotlib.pyplot as plt

import imgsolver.expr2seg_img as e2s
import keras
import numpy as np
from typing import Callable


class ImgSolver:
    
    MODEL_IDX: int = 0
    LUT_IDX: int = 1

    def __init__(self, version: int = 4):
        self.category_model_path = f'models/category_class_v{version}'
        self.digit_model_path = f'models/digit_class_v{version}'
        self.operator_model_path = f'models/operator_class_v{version}'
        self.paren_model_path = f'models/paren_class_v{version}'
        
        self.model_ext = '.model.keras'
        self.indices_ext = '_indices.txt'
        
        self.models:dict[str, tuple[keras.models.Sequential, dict[str, str]]] = {}
        
        model_paths = [self.category_model_path+self.model_ext, self.digit_model_path+self.model_ext, self.operator_model_path+self.model_ext, self.paren_model_path+self.model_ext]
        indices_paths = [self.category_model_path+self.indices_ext, self.digit_model_path+self.indices_ext, self.operator_model_path+self.indices_ext, self.paren_model_path+self.indices_ext]

        if all(map(lambda x: os.path.exists(x), model_paths)):
            self.models['category'] = [keras.models.load_model(self.category_model_path+self.model_ext), None]
            self.models['digit'] = [keras.models.load_model(self.digit_model_path+self.model_ext), None]
            self.models['operator'] = [keras.models.load_model(self.operator_model_path+self.model_ext), None]
            self.models['paren'] = [keras.models.load_model(self.paren_model_path+self.model_ext), None]
        else:
            raise Exception("Some models are missing")

        if all(map(lambda x: os.path.exists(x), indices_paths)):
            self.models['category'][ImgSolver.LUT_IDX] = self.read_indices(self.category_model_path+self.indices_ext)            
            self.models['digit'][ImgSolver.LUT_IDX] = self.read_indices(self.digit_model_path+self.indices_ext)            
            self.models['operator'][ImgSolver.LUT_IDX] = self.read_indices(self.operator_model_path+self.indices_ext)            
            self.models['paren'][ImgSolver.LUT_IDX] = self.read_indices(self.paren_model_path+self.indices_ext)            
        else:
            raise Exception('Missing indices files')


    def eval(self, img, class2charCvt: Callable[[str], str]):
        seg_xywh = e2s.expr2segm_img(img)
        sorted_seg = sorted(seg_xywh, key=lambda x: x[1])
                
        expression_chars = []
        
        for segment, x, y, w, h in sorted_seg:
            segment = segment.astype('float32')
            plt.imshow(segment)
            plt.show()
            segment = np.expand_dims(segment, axis=0)
            category_pred = self.models['category'][ImgSolver.MODEL_IDX].predict(segment)
            category = self.models['category'][ImgSolver.LUT_IDX][np.argmax(category_pred)]           
            print('Category distribution v4: ', category_pred)
            print('Category prediction v4: ', category, '\n')
            
            prediction = None
            pred_dist = self.models[category][ImgSolver.MODEL_IDX].predict(segment)
            prediction = self.models[category][ImgSolver.LUT_IDX][np.argmax(pred_dist)]            
            
            print('Prediction distribution: ', pred_dist)
            prediction = class2charCvt(prediction)
            print('Prediction: ', prediction)
            print('Confidence: ', np.max(pred_dist)*np.max(category_pred))
            
            
            expression_chars.append((prediction, x, y, w, h))
        return expression_chars

    def read_indices(self, path: str):
        with open(path, 'r') as fp:
            return eval(fp.read())