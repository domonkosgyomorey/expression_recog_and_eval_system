import os
import keras
import numpy as np
import sympy
from imgsolver import expr2seg_img as e2s
import matplotlib.pylab as plt
from imgsolver.model_trainer.my_cnn_models.cnn_util import custom_loss

class ImgSolver:
    
    MODEL_IDX: int = 0
    LUT_IDX: int = 1

    def __init__(self, models_path, model_version: int = 4, verbose: bool = False):
        self.verbose = verbose
        
        self.category_model_path = models_path + f'/category_class_v{model_version}2024-11-09'
        self.digit_model_path = models_path + f'/digit_class_v{model_version}2024-11-09'
        self.operator_model_path = models_path + f'/operator_class_v{model_version+1}2024-11-09'
        self.paren_model_path = models_path + f'/paren_class_v{model_version}2024-11-09'
        
        self.model_ext = '.model.keras'
        self.indices_ext = '_indices.txt'
        
        self.models: dict[str, tuple[keras.models.Sequential, dict[str, str]]] = {}
        
        model_paths = [
            self.category_model_path + self.model_ext,
            self.digit_model_path + self.model_ext,
            self.operator_model_path + self.model_ext,
            self.paren_model_path + self.model_ext
        ]

        
        indices_paths = [
            self.category_model_path + self.indices_ext,
            self.digit_model_path + self.indices_ext,
            self.operator_model_path + self.indices_ext,
            self.paren_model_path + self.indices_ext
        ]


        if all(map(os.path.exists, model_paths)):
            self.models['category'] = [keras.models.load_model(self.category_model_path + self.model_ext, custom_objects={'custom_loss': custom_loss}), None]
            self.models['digit'] = [keras.models.load_model(self.digit_model_path + self.model_ext, custom_objects={'custom_loss': custom_loss}), None]
            self.models['operator'] = [keras.models.load_model(self.operator_model_path + self.model_ext, custom_objects={'custom_loss': custom_loss}), None]
            self.models['paren'] = [keras.models.load_model(self.paren_model_path + self.model_ext, custom_objects={'custom_loss': custom_loss}), None]
        else:
            raise Exception("Some models are missing")

        if all(map(os.path.exists, indices_paths)):
            self.models['category'][ImgSolver.LUT_IDX] = self.read_indices(self.category_model_path + self.indices_ext)
            self.models['digit'][ImgSolver.LUT_IDX] = self.read_indices(self.digit_model_path + self.indices_ext)
            self.models['operator'][ImgSolver.LUT_IDX] = self.read_indices(self.operator_model_path + self.indices_ext)
            self.models['paren'][ImgSolver.LUT_IDX] = self.read_indices(self.paren_model_path + self.indices_ext)
        else:
            raise Exception('Missing indices files')

    def eval(self, img) -> tuple[str, str]:
        expression = []
        try:
            seg_xywh = e2s.expr2segm_img(img)
            sorted_seg = sorted(seg_xywh, key=lambda x: (x[1]+x[3])//2)
            
            segments = [np.expand_dims(segment.astype('float32'), axis=0) for segment, x, y, w, h, area in sorted_seg]
            batch_segments = np.vstack(segments)
            
            category_preds = self.models['category'][ImgSolver.MODEL_IDX].predict(batch_segments, verbose=0)
            category_indices = np.argmax(category_preds, axis=1)
            categories = [self.models['category'][ImgSolver.LUT_IDX][index] for index in category_indices]
            
            if self.verbose:
                print('Category predictions:', categories)
            
            expression_chars = []
            
            for i, (category, segment_info) in enumerate(zip(categories, sorted_seg)):
                segment, x, y, w, h, area = segment_info
                pred_dist = self.models[category][ImgSolver.MODEL_IDX].predict(np.expand_dims(segment.astype('float32'), axis=0), verbose=0)
                prediction = self.models[category][ImgSolver.LUT_IDX][np.argmax(pred_dist)]
                
                if self.verbose:
                    print('Prediction distribution for', category, ':', pred_dist)
                
                prediction = self.cvt_str2op(prediction)
                
                if self.verbose:
                    print('Prediction:', prediction)
                    print('Confidence:', np.max(pred_dist) * np.max(category_preds[i]))
                
                expression_chars.append((prediction, x, y, w, h, area))

            # Check if a segment is too small
            for i in range(len(expression_chars)):
                if expression_chars[i][5] < 10:
                    s, x, y, w, h, a = expression_chars[i]
                    s = '.'
                    expression_chars[i] = (s, x, y, w, h, a) 
            
            expression = "".join(map(lambda x: x[0], expression_chars))
            if self.verbose:
                print("[LOG]: ImgSolver, found:", expression)
            
            #TODO: Analize the predicted segments to find fraction, and exponent
            return (expression, sympy.simplify(expression))
        except Exception as e:
            return (f"Error: Cannot evaulate this: "+str(expression), None)
            
    def cvt_str2op(self, predicted_class: str) -> str:
        if predicted_class == "times":
            return "*"
        elif predicted_class == "div":
            return "/"
        elif predicted_class == ',':
            return "."
        return predicted_class
    
    def read_indices(self, path: str):
        with open(path, 'r') as fp:
            return eval(fp.read())
