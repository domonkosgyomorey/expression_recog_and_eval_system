from my_cnn_models import cnn_util as cnnu

cnnu.train_model(cnnu.create_model_v2(2), 'dataset/binary_class', 'models/num_sym_bin_classification_v2.model.keras', 10, True, True)