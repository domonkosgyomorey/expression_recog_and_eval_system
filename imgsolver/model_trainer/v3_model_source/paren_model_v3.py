import imgsolver.model_trainer.my_cnn_models.cnn_util as cnnu
models = {
    'paren_class_v3' : (cnnu.create_model_v3(2), 'dataset/paren'),
}
cnnu.train_models(models, '../../models/', 20, enable_plotting=False)
    