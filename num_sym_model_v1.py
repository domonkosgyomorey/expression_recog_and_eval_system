from my_cnn_models import cnn_util as cnnu

cnnu.create_and_train_w_aug(cnnu.create_model_v1(2), 'dataset/binary_class', 'num_sym_bin_classification_v1.model.keras', 10, cnnu.CATEGORICAL_CLASS_MODE)
