import my_cnn_models.cnn_util as cnnu
models = {
    'operator_class_v4' : (cnnu.create_model_v4(5), 'dataset/trig_log'),
}
cnnu.train_models(models, 'models/', 20)
    