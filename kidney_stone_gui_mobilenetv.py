import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("kidney_stone_mobilenetv2.h5")

# Predict function
def predict_image(filepath):
    img = image.load_img(filepath, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model.predict(img)[0][0]
    return "Kidney Stone Detected" if prediction < 0.5 else "No Stone Detected"

# GUI Functions /
def upload_action():
    global img_path
    img_path = filedialog.askopenfilename()
    img = Image.open(img_path)
    img = img.resize((250, 250))
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img
    result_label.config(text="Image Uploaded. Click Analyze.")

def analyze_action():
    if img_path:
        result = predict_image(img_path)
        result_label.config(text=result)

def clear_action():
    panel.configure(image='')
    panel.image = None
    result_label.config(text="")

# Create GUI
root = tk.Tk()
root.title("Kidney Stone Detection GUI")
root.geometry("500x600")
root.configure(bg="lightblue")

heading = tk.Label(root, text="Kidney Stone Detection", font=("Arial", 20, "bold"), bg="lightblue")
heading.pack(pady=10)

panel = tk.Label(root)
panel.pack()

upload_btn = tk.Button(root, text="Upload Image", command=upload_action, font=("Arial", 14))
upload_btn.pack(pady=10)

analyze_btn = tk.Button(root, text="Analyze Image", command=analyze_action, font=("Arial", 14))
analyze_btn.pack(pady=10)

clear_btn = tk.Button(root, text="Clear", command=clear_action, font=("Arial", 14))
clear_btn.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 16), bg="lightblue")
result_label.pack(pady=10)

img_path = None
root.mainloop()
