import keras
import my_cnn_models.cnn_util as cnnu
t_g, v_g = cnnu.create_train_validation_generator('dataset/binary_class/operator', rescale=1/255, target_size=(45, 45), class_mode=cnnu.CATEGORICAL_CLASS_MODE)

model : keras.models.Sequential = keras.models.load_model('symbol_classification_v3.model.keras')
print(model.evaluate(v_g))