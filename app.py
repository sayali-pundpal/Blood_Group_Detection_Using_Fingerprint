import tkinter as tk
from tkinter import filedialog, messagebox
from .prediction import BloodGroupPredictor

class FingerprintApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Fingerprint Blood Group Predictor")
        self.root.geometry("400x300")
        
        self.predictor = BloodGroupPredictor(model_path)
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create GUI widgets"""
        self.label = tk.Label(
            self.root, 
            text="Upload Fingerprint Image (BMP Only)", 
            font=("Arial", 14))
        self.label.pack(pady=20)
        
        self.button = tk.Button(
            self.root, 
            text="Upload Image", 
            command=self.upload_image)
        self.button.pack()
        
        self.result_label = tk.Label(
            self.root, 
            text="", 
            font=("Arial", 12))
        self.result_label.pack(pady=10)
    
    def upload_image(self):
        """Handle image upload and prediction"""
        file_path = filedialog.askopenfilename(filetypes=[("BMP Files", "*.bmp")])
        if file_path:
            try:
                blood_group = self.predictor.predict(file_path)
                self.result_label.config(text=f"Predicted Blood Group: {blood_group}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

def run_app(model_path="models/fingerprint_bloodgroup_model.h5"):
    """Run the GUI application"""
    root = tk.Tk()
    app = FingerprintApp(root, model_path)
    root.mainloop()
