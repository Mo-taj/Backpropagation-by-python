import cv2
import tkinter as tk
from PIL import Image, ImageTk
import tkinter.filedialog as tkFileDialog
import numpy as np

# Global variables
counter = 0
P = []
T = []
b = []
weights = np.array([])
bias = np.array([])
image = None
text = (" ")


def open_image():
    global image
    path = tkFileDialog.askopenfilename(filetypes=[("Image Files", ".jpg .png .gif")])
    if path:
        im = Image.open(path)
        tkimage = ImageTk.PhotoImage(im)
        image = tkimage
        label1.config(image=tkimage)
        label1.image = tkimage
        neural(path)


def training():
    global weights, T, P, b
    S = 1
    for i in range(10):
        P.append(flatten(cv2.imread(f"data2/cat.{i}.jpg", cv2.IMREAD_GRAYSCALE)))
        T.append([1 for _ in range(S)])
        P.append(flatten(cv2.imread(f"data2/dog.{i}.jpg", cv2.IMREAD_GRAYSCALE)))
        T.append([0 for _ in range(S)])
    P = np.array(P)
    T = np.array(T)
    numP = len(P)
    R = len(P[0])
    weights = np.zeros([S, R])
    b = np.zeros([S, 1])
    for epoch in range(100):
        for index in range(numP):
            p = np.array([P[index]]).transpose()
            n = (np.dot(weights, p) + b)
            a = [[1] if el[0] >= 0 else [0] for el in n]
            e = T[index].reshape(S, 1) - a
            weights += np.dot(e, p.transpose())
            b += e
    print("Training Completed.")
    label2.config(text="Model Trained", fg="green")
    button1.pack_forget()  # Hide the "Train Model" button
    button2.pack_forget()  # Hide the "Exit" button
    button3.pack(side="left", padx=10)  # Show the "Select Image" button
    button4.pack(side="right", padx=10)  # Show the "Return" button


def return_to_main():
    button3.pack_forget()  # Hide the "Select Image" button
    button4.pack_forget()  # Hide the "Return" button
    button1.pack(side="left", padx=10)  # Show the "Train Model" button
    button2.pack(side="right", padx=10)  # Show the "Exit" button


def flatten(image):
    new_image = []
    for row in image:
        for el in row:
            new_image.append(el)
    return new_image


def neural(path):
    global image, weights, text, b
    if not np.any(weights) or not np.any(b):
        label2.config(text="Model not trained!", fg="red")
        return
    p = np.array(flatten(cv2.imread(path, cv2.IMREAD_GRAYSCALE)))
    p = p.transpose()
    n = np.dot(weights, p) + b
    confidence = 1 / (1 + np.exp(-n))  # Sigmoid function to get confidence level
    confidence_str = f"Confidence: {confidence[0][0]:.2f}"
    text = "The animal type is: Cat" if n[0][0] >= 0 else "The animal type is: Dog"
    label2.config(text=f"{text}\n{confidence_str}")
    label2.text = text


# Create a tkinter window
window = tk.Tk()
window.title("Image Classification")
window.geometry("700x650")
window.configure(bg="#24293E")  # Set background color

# Load default image
image = Image.open("none.jpg")
photo = ImageTk.PhotoImage(image)

# Create labels for images
label1 = tk.Label(window, image=photo, bg="#24293E", borderwidth=2, relief="solid")
label1.pack(pady=10)

# Create a label
label2 = tk.Label(window, text=text, bg="#24293E", fg="#ff9b54", font=("Helvetica", 16))
label2.pack(pady=10, anchor="s")

# Create buttons
button_frame = tk.Frame(window, bg="#24293E")
button_frame.pack(side="bottom", pady=10)
button1 = tk.Button(button_frame, text="Training", command=lambda: training(), bg="#8EBBFF", fg="#24293E",
                    padx=20, pady=10, font=("Helvetica", 14))
button2 = tk.Button(button_frame, text="Exit", command=window.destroy, bg="#FF7F51", fg="#24293E",
                    padx=20, pady=10, font=("Helvetica", 14))
button3 = tk.Button(button_frame, text="Select Image", command=lambda: open_image(), bg="#CE4257", fg="#24293E",
                    padx=20, pady=10, font=("Helvetica", 14))
button4 = tk.Button(button_frame, text="Return", command=lambda: return_to_main(), bg="#FF7F51", fg="#24293E",
                    padx=20, pady=10, font=("Helvetica", 14))
button1.pack(side="left", padx=10)
button2.pack(side="right", padx=10)

# Start the tkinter event loop
window.mainloop()
