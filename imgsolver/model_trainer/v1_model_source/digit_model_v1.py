import imgsolver.model_trainer.my_cnn_models.cnn_util as cnnu
models = {
    'digit_class_v1' : (cnnu.create_model_v1(10), 'dataset/digit'),
}
cnnu.train_models(models, '../../models/', 20, enable_plotting=False)
    