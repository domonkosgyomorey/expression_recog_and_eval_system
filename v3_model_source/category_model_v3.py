import my_cnn_models.cnn_util as cnnu
models = {
    'category_class_v3' : (cnnu.create_model_v3(3), 'dataset'),
}
cnnu.train_models(models, 'models/', 20)
    