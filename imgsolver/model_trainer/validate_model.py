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

suffix = '../../models/'
creation_date = '2024-11-09'
model_and_dataset = [
    (suffix+'operator_class_v5'+creation_date+'.model.keras', 'dataset/operator'),
    (suffix+'digit_class_v4'+creation_date+'.model.keras', 'dataset/digit'),
    (suffix+'paren_class_v4'+creation_date+'.model.keras', 'dataset/paren'),
    (suffix+'category_class_v4'+creation_date+'.model.keras', 'dataset/'),
]

model:models.Sequential = models.load_model(model_and_dataset[OPERATOR][MODEL_PATH])
validate_models(model, model_and_dataset[OPERATOR][DATASET_PATH])