import my_cnn_models.cnn_util as cnnu
models = {
    'paren_class_v2' : (cnnu.create_model_v2(2), 'dataset/paren'),
}
cnnu.train_models(models, '../models/', 20)
    