import os
from keras import models

from my_cnn_models.cnn_util import validate_models

prefix = '../../models/'
file_names = ['category_class_v42024-11-09.model.keras', 'digit_class_v42024-11-09.model.keras', 'operator_class_v42024-11-09.model.keras', 'paren_class_v42024-11-09.model.keras']
datasets = ['dataset', 'dataset/digit', 'dataset/operator', 'dataset/paren']
model_metric_logs_path = './model_metric_logs/'
os.makedirs(os.path.dirname(model_metric_logs_path), exist_ok=True)

for model_name, ds_path in zip(file_names, datasets):
    validate_models(
        models.load_model(prefix+model_name),
        ds_path,
        model_metric_logs_path+model_name+
        '_metrics.xlsx'
    )