import my_cnn_models.cnn_util as cnnu
models = {
    'operator_class_v1' : (cnnu.create_model_v1(5), 'dataset/operator'),
}
cnnu.train_models(models, 'models/', 20)
    