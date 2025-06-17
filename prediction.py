import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

class BloodGroupPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = None
        self.img_size = (128, 128)
    
    def load_class_names(self, generator):
        """Load class names from training generator"""
        self.class_names = list(generator.class_indices.keys())
    
    def predict(self, image_path):
        """
        Predict blood group from fingerprint image
        """
        if not image_path.lower().endswith(".bmp"):
            raise ValueError("Only BMP images are supported.")
        
        img = load_img(image_path, target_size=self.img_size, color_mode='rgb')
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = self.model.predict(img_array)
        
        if self.class_names is None:
            raise ValueError("Class names not loaded. Please provide training generator.")
        
        return self.class_names[np.argmax(prediction)]
