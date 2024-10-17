from my_cnn_models import cnn_util as cnnu

cnnu.create_and_train_w_aug(cnnu.create_model_v3(2), 'dataset/binary_class', 'num_sym_bin_classification_v3.model.keras', 10, cnnu.CATEGORICAL_CLASS_MODE)
