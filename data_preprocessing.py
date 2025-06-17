import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset(dataset_path, img_size=(128, 128), batch_size=32, validation_split=0.2):
    """
    Load and preprocess the dataset using ImageDataGenerator
    """
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)
    
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')
    
    val_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')
    
    return train_generator, val_generator
