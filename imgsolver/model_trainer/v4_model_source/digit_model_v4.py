import my_cnn_models.cnn_util as cnnu
models = {
    'digit_class_v4' : (cnnu.create_model_v4(10), 'dataset/digit'),
}
cnnu.train_models(models, '../../models/', 20)
    