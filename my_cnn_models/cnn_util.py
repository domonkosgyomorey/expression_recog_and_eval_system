from keras import callbacks
import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras import layers, models
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau 
import cv2
import numpy as np
import keras
import livelossplot as llp
import os

CATEGORICAL_CLASS_MODE = 'categorical'

def create_checkpoint(file_path):
    return callbacks.ModelCheckpoint(file_path, save_best_only=True, monitor='val_loss', mode='min', save_weights_only=False, save_freq='epoch')

def create_train_validation_generator(dataset_path, rescale=1.0/255.0,
                                      validation_split=0.3,
                                      target_size=(224, 224),
                                      batch_size=32,
                                      class_mode='categorical'):

    data_gen = ImageDataGenerator(rescale=rescale, validation_split=validation_split)

    train_generator = data_gen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode=class_mode,
        subset='training'
    )

    validation_generator = data_gen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode=class_mode,
        subset='validation')

    return train_generator, validation_generator

def thickening_line(img):
    kernel = np.ones((2, 2))
    return cv2.erode(img, kernel)    

def create_train_validation_generator_augmented(dataset_path, rescale=1.0/255.0,
                                      validation_split=0.3,
                                      target_size=(45, 45),
                                      batch_size=32,
                                      class_mode='categorical'):
    data_gen = ImageDataGenerator(
        rescale=rescale,
        validation_split=validation_split,
        rotation_range=10,
        preprocessing_function=thickening_line
    )
    train_generator = data_gen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode=class_mode,
        subset='training'
    )

    validation_generator = data_gen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode=class_mode,
        subset='validation')

    return train_generator, validation_generator
    
def save_class_indices(file_path, class_indices):
    with open(file_path, 'w') as f:
        class_indices = {y: x for x, y in class_indices.items()}
        f.write(str(class_indices))

def write_result_statistics(model, validation_generator, class_indices):
    val_loss, val_acc = model.evaluate(validation_generator)
    print(f'Val accuracy: {val_acc}')
    print(f'Val loss: {val_loss}')

def plot_history(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def create_model_v1(num_of_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(45, 45, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_of_classes, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_model_v2(num_of_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(45, 45, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_of_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    y = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, kernel_size, padding="same")(y)
    y = layers.BatchNormalization()(y)

    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same")(x)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.Add()([y, shortcut])
    y = layers.Activation("relu")(y)

    return y

def create_model_v3(num_of_classes):
    inputs = layers.Input(shape=(45, 45, 1))

    x = layers.Conv2D(32, 3, strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = residual_block(x, 32)
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_of_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_model_v4(num_classes) -> models.Sequential:
    model = models.Sequential([
        layers.Input(shape=(45,45,1)),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
    
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
    
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, dataset_path, model_name, epoch, do_augmentation=False, enable_plotting=True):
    model.summary()
    
    train_generator = None
    validation_generator = None
    
    if do_augmentation:
        train_generator, validation_generator = create_train_validation_generator(dataset_path, class_mode=
                                                                                  CATEGORICAL_CLASS_MODE, target_size=(45, 45))
    else:
        train_generator, validation_generator = create_train_validation_generator_augmented(dataset_path, class_mode=CATEGORICAL_CLASS_MODE, target_size=(45, 45))
            
    save_class_indices(model_name+'_indices.txt', train_generator.class_indices)
    
    history = None
    if enable_plotting:
        plot_loss = llp.PlotLossesKeras()
        
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            epochs=epoch,
            callbacks=[create_checkpoint(model_name), plot_loss]
        )        
    else:
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            epochs=epoch,
            callbacks=[create_checkpoint(model_name)]
        )
        
    write_result_statistics(model, validation_generator, validation_generator.class_indices)
    plot_history(history)
    
    return model

def train_models(models_and_names : dict[str, (models.Sequential, str)], path, epoch, enable_plotting=True):
    for name, (model, dataset_path) in models_and_names.items():

        print(f"Loading dateset for {name} model")

        train_generator, validation_generator = create_train_validation_generator(dataset_path, class_mode=
            CATEGORICAL_CLASS_MODE, target_size=(45, 45))

        save_class_indices(path+name + '_indices.txt', train_generator.class_indices)

        print(f"Dataset loaded for {name} model")

        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        plot_loss = llp.PlotLossesKeras()

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        print(f"Training {name} model")

        model_path = path+name+'.model'+'.keras'
        
        if os.path.exists(model_path):
            print('Model is already exists')
            model:models.Sequential = models.load_model(model_path)

        model.fit(train_generator,
                  steps_per_epoch=train_generator.samples // train_generator.batch_size,
                  validation_data=validation_generator,
                  validation_steps=validation_generator.samples // validation_generator.batch_size,
                  epochs=epoch,
                  callbacks=[
                      EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                      ReduceLROnPlateau(
                          monitor='val_loss',
                          factor=0.5,
                          patience=3,
                          min_lr=0.00001
                      ),
                      create_checkpoint(model_path),
                      plot_loss])
        
        return model