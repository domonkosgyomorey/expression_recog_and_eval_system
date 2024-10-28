import my_cnn_models.cnn_util as cnnu
models = {
    'paren_class_v4' : (cnnu.create_model_v4(2), 'dataset/paren'),
}
cnnu.train_models(models, '../models/', 20)
    