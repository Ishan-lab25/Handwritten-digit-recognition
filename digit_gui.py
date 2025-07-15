import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("model.keras")

# Create the GUI app
root = tk.Tk()
root.title("Digit Recognizer")

canvas = tk.Canvas(root, width=200, height=200, bg="white")
canvas.pack()

# Image object to store drawing
image = Image.new("L", (200, 200), color="white")
draw = ImageDraw.Draw(image)

# Drawing with mouse
def draw_digit(event):
    x, y = event.x, event.y
    r = 8
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
    draw.ellipse([x-r, y-r, x+r, y+r], fill="black")

canvas.bind("<B1-Motion>", draw_digit)

# Predict function
def predict_digit():
    img = image.resize((28, 28))
    img = ImageOps.invert(img)
    img = np.array(img).astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    pred = model.predict(img)
    digit = np.argmax(pred)
    result_label.config(text=f"Prediction: {digit}")

# Add button and label
predict_btn = tk.Button(root, text="Predict", command=predict_digit)
predict_btn.pack()

result_label = tk.Label(root, text="Draw a digit and click Predict")
result_label.pack()

root.mainloop()
