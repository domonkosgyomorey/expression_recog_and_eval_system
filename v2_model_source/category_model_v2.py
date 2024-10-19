import my_cnn_models.cnn_util as cnnu
models = {
    'category_class_v2' : (cnnu.create_model_v2(4), 'dataset'),
}
cnnu.train_models(models, 'models/', 20)
    