from os import listdir
from os.path import isfile, join
import datetime

from keras import models
from my_cnn_models.cnn_util import validate_models

prefix = '../models/'
file_names = ['category_class_v4.model.keras', 'digit_class_v4.model.keras', 'operator_class_v4.model.keras', 'paren_class_v4.model.keras']
datasets = ['dataset', 'dataset/digit', 'dataset/operator', 'dataset/paren']

for model_name, ds_path in zip(file_names, datasets):
    validate_models(models.load_model(prefix+model_name), ds_path, model_name+'_metrics_'+str(datetime.datetime.now().date()).replace('-', '_')+'.xlsx')