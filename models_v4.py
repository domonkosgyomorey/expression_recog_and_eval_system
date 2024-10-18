import my_cnn_models.cnn_util as cnnu

digit_class = cnnu.create_model_v4(10)
operator_class = cnnu.create_model_v4(5)
paren_class = cnnu.create_model_v4(2)
trig_log_class = cnnu.create_model_v4(4)

category_class = cnnu.create_model_v4(4)

models = {
    'digit_class': (digit_class, 'dataset/binary_class/digit'),
    'operator_class': (operator_class, 'dataset/binary_class/operator'),
    'paren_class': (paren_class, 'dataset/binary_class/paren'),
    'trig_log_class': (trig_log_class, 'dataset/binary_class/trig_log'),
    'category_class': (category_class, 'dataset/binary_class'),
}

cnnu.train_models(models, 20)