import imgsolver.model_trainer.my_cnn_models.cnn_util as cnnu
models = {
    'category_class_v1' : (cnnu.create_model_v1(3), 'dataset'),
}
cnnu.train_models(models, '../../models/', 20, enable_plotting=False)
    