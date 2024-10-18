from my_cnn_models import cnn_util as cnnu

cnnu.train_model(cnnu.create_model_v3(11), 'dataset/binary_class/digit', 'models/number_classification_v3.model.keras', 10, True, True)
