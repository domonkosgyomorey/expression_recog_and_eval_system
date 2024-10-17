from my_cnn_models import cnn_util as cnnu

cnnu.create_and_train_w_aug(cnnu.create_model_v2(11), 'dataset/binary_class/digit', 'number_classification_v2.model.keras', 10, cnnu.CATEGORICAL_CLASS_MODE)
