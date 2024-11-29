import os
from keras import models

from my_cnn_models.cnn_util import validate_models

version = 1
date = '2024-11-29'
prefix = '../../models/'
file_names = [f'category_class_v{version}{date}.model.keras', f'digit_class_v{version}{date}.model.keras', f'operator_class_v{version}{date}.model.keras', f'paren_class_v{version}{date}.model.keras']
datasets = ['test_dataset', 'test_dataset/digit', 'test_dataset/operator', 'test_dataset/paren']
model_metric_logs_path = './model_metric_logs/'
os.makedirs(os.path.dirname(model_metric_logs_path), exist_ok=True)

for model_name, ds_path in zip(file_names, datasets):
    validate_models(models.load_model(prefix+model_name), ds_path, model_metric_logs_path+model_name+ '_metrics.xlsx')