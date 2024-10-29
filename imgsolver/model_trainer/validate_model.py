import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 

from my_cnn_models.cnn_util import validate_models
from keras import models

OPERATOR = 0
DIGIT = 1
PAREN = 2
CATEGORY = 3

MODEL_PATH = 0
DATASET_PATH = 1

ext = '_class_v4.model.keras'
model_and_dataset = [
    ('../models/operator'+ext, 'dataset/operator'),
    ('../models/digit'+ext, 'dataset/digit'),
    ('../models/paren'+ext, 'dataset/paren'),
    ('../models/category'+ext, 'dataset/')
]

model:models.Sequential = models.load_model(model_and_dataset[CATEGORY][MODEL_PATH])

validate_models(model, model_and_dataset[CATEGORY][DATASET_PATH], 'asd')