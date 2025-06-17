import tensorflow as tf
from .data_preprocessing import load_dataset
from .model import create_model
from .visualization import plot_results

def train_model(dataset_path, img_size=(128, 128), batch_size=32, epochs=50):
    """
    Train the model and save it
    """
    # Load dataset
    train_generator, val_generator = load_dataset(
        dataset_path, 
        img_size=img_size, 
        batch_size=batch_size)
    
    # Create model
    model = create_model(
        input_shape=(img_size[0], img_size[1], 3),
        num_classes=len(train_generator.class_indices))
    
    # Train model
    history = model.fit(
        train_generator, 
        validation_data=val_generator, 
        epochs=epochs)
    
    # Save model
    model.save("models/fingerprint_bloodgroup_model.h5")
    
    # Plot results
    plot_results(history)
    
    return model, history
