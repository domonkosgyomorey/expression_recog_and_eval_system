import imgsolver.model_trainer.my_cnn_models.cnn_util as cnnu
models = {
    'operator_class_v5' : (cnnu.create_model_v5(4), 'dataset/operator'),
}
cnnu.train_models(models, '../../models/', 20, enable_plotting=False)
    