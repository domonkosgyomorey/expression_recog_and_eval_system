from keras import callbacks
import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator, DirectoryIterator
from keras import layers, models
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau 
import cv2
import numpy as np
import keras
import livelossplot as llp
import os
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve, 
                           average_precision_score)
import seaborn as sns
import pandas as pd

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
        subset='training',
        shuffle=True
    )

    validation_generator = data_gen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode=class_mode,
        subset='validation',
        shuffle=False   
    )
    
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

def compute_class_weights(data_generator: DirectoryIterator):
    total_samples = len(data_generator.classes)
    class_counts = np.bincount(data_generator.classes)
    return {i: total_samples / (len(class_counts)*count) for i, count in enumerate(class_counts)}
    
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

def train_model(model: models.Model, dataset_path, model_name, epoch, do_augmentation=False, enable_plotting=True):
    model.summary()
    
    train_generator = None
    validation_generator = None
    
    if do_augmentation:
        train_generator, validation_generator = create_train_validation_generator(dataset_path, class_mode=
                                                                                  CATEGORICAL_CLASS_MODE, target_size=(45, 45))
    else:
        train_generator, validation_generator = create_train_validation_generator_augmented(dataset_path, class_mode=CATEGORICAL_CLASS_MODE, target_size=(45, 45))
            
    save_class_indices(model_name+'_indices.txt', train_generator.class_indices)
    plot_loss = llp.PlotLossesKeras()
    
    history = None
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.0000001
        ),
        create_checkpoint(model_name)
    ]
    
    if enable_plotting:
            callbacks.append(plot_loss)
            
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=epoch,
        class_weight=compute_class_weights(train_generator),
        callbacks=callbacks
    )
        
    write_result_statistics(model, validation_generator, validation_generator.class_indices)
    plot_history(history)
    
    return model

def train_models(models_and_names : dict[str, (models.Sequential, str)], path, epoch, enable_plotting=True):
    for name, (model, dataset_path) in models_and_names.items():
        model.summary()
        
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

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.0000001
            ),
            create_checkpoint(model_path),
        ]
        if enable_plotting:
            callbacks.append(plot_loss)
        
        model.fit(train_generator,
                  steps_per_epoch=train_generator.samples // train_generator.batch_size,
                  validation_data=validation_generator,
                  validation_steps=validation_generator.samples // validation_generator.batch_size,
                  epochs=epoch,
                  callbacks=callbacks
                  )
        
    return model
    
def validate_models(model, dataset_path, metric_path):
    model:keras.Sequential = model 
    _, validation_generator = create_train_validation_generator(dataset_path, class_mode=
        CATEGORICAL_CLASS_MODE, target_size=(45, 45))
    val_results = model.evaluate(validation_generator, verbose=1)
    metric_names = model.metrics_names
    print("\nValidation metrics:")
    for name, value in zip(metric_names, val_results):
        print(f'{name}: {value:.4f}')
    
    metrics_df = pd.DataFrame(val_results, index=metric_names, columns=["Value"])
    
    print("\nPredictions")
    validation_generator.reset()
    y_pred_proba = model.predict(validation_generator, verbose=1)
    
    validation_generator.reset()
    y_true = []
    for _ in range(len(validation_generator)):
        _, batch_labels = next(validation_generator)
        y_true.extend(batch_labels)
    y_true = np.array(y_true)
    
    if y_true.shape[-1] > 1:
        y_true_classes = np.argmax(y_true, axis=1)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
    else:
        y_true_classes = y_true
        y_pred_classes = (y_pred_proba > 0.5).astype(int)
        
    print("Classification riport:")
    class_report = classification_report(y_true_classes, y_pred_classes)
    print(class_report)
    
    report_data = classification_report(y_true_classes, y_pred_classes, output_dict=True)
    report_df = pd.DataFrame(report_data).transpose()

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    cm_df = pd.DataFrame(cm, index=[f"True_{i}" for i in range(cm.shape[0])],
                         columns=[f"Pred_{i}" for i in range(cm.shape[1])])
    
    with pd.ExcelWriter(metric_path, engine='openpyxl') as writer:
        metrics_df.to_excel(writer, sheet_name='Validation Metrics')
        report_df.to_excel(writer, sheet_name='Classification Report')
        cm_df.to_excel(writer, sheet_name='Confusion Matrix')
    
    def plot_learning_curves():
        try:
            history = model.history.history
            if history:
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(history['accuracy'], label='Training')
                plt.plot(history['val_accuracy'], label='Validation')
                plt.title('Model Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(history['loss'], label='Training')
                plt.plot(history['val_loss'], label='Validation')
                plt.title('Model Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                
                plt.tight_layout()
                plt.show()
        except:
            print("No model history")
    
    def plot_confusion_matrix():
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Real')
        plt.xlabel('Predicted')
        plt.show()
        
        plt.figure(figsize=(10, 8))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('Real')
        plt.xlabel('Predicted')
        plt.show()
    
    def plot_roc_curves():
        plt.figure(figsize=(10, 8))
        
        if y_true.shape[-1] > 1:
            n_classes = y_true.shape[-1]
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        else:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.show()
    
    def plot_precision_recall_curve():
        plt.figure(figsize=(10, 8))
        
        if y_true.shape[-1] > 1:
            n_classes = y_true.shape[-1]
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_proba[:, i])
                avg_precision = average_precision_score(y_true[:, i], y_pred_proba[:, i])
                plt.plot(recall, precision, label=f'Class {i} (AP = {avg_precision:.2f})')
        else:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)
            plt.plot(recall, precision, label=f'PR (AP = {avg_precision:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall')
        plt.legend(loc="lower left")
        plt.show()
    
    plot_learning_curves()
    plot_confusion_matrix()
    plot_roc_curves()
    plot_precision_recall_curve()
    
    metrics = {
        'base_metrics': dict(zip(metric_names, val_results)),
        'classification_report': class_report,
        'confusion_matrix': confusion_matrix(y_true_classes, y_pred_classes)
    }
    
    if y_true.shape[-1] > 1:
        metrics['roc_auc'] = {
            f'class_{i}': auc(
                *roc_curve(y_true[:, i], y_pred_proba[:, i])[:2]
            ) for i in range(y_true.shape[-1])
        }
    else:
        metrics['roc_auc'] = auc(*roc_curve(y_true, y_pred_proba)[:2])