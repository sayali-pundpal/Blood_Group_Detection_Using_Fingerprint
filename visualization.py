import matplotlib.pyplot as plt

def plot_results(history):
    """
    Plot training and validation accuracy/loss
    """
    plt.figure(figsize=(10,5))
    
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title("Model Accuracy")
    
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Model Loss")
    
    plt.tight_layout()
    plt.show()
