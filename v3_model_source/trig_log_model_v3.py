import my_cnn_models.cnn_util as cnnu
models = {
    'trig_log_class_v3' : (cnnu.create_model_v3(4), 'dataset'),
}
cnnu.train_models(models, 'models/', 20)
    