import my_cnn_models.cnn_util as cnnu
models = {
    'category_class_v4' : (cnnu.create_model_v4(4), 'dataset'),
}
cnnu.train_models(models, 'models/', 20)
    