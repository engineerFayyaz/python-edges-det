import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import imutils
import tensorflow as tf
import numpy as np
import os

# Create the main application window
root = tk.Tk()
root.title("Image Analysis Application")
root.geometry("800x600")  # Set the window size

# Configure background color
root.configure(bg="#f0f0f0")  # Use your preferred background color

# Global variables for image and model
img_path = None
model = None

# Load the MobileNet SSD model for object detection
def load_model():
    global model
    # Load your AI model here
    # model = tf.saved_model.load("models")
    model_dir = os.path.abspath("models")

# Function to load an image from the computer
def load_image():
    global img_path
    img_path = filedialog.askopenfilename(title="Select an image file",
                                         filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")])
    if not img_path:
        messagebox.showinfo("Information", "No image selected. Please select an image.")
    else:
        process_image()

# Function to handle the "Select Image" button click
def select_image():
    load_image()

# Function to capture an image from the camera
def capture_image():
    cap = cv2.VideoCapture(0)  # Open the default camera

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open camera.")
        return

    ret, frame = cap.read()  # Capture a frame
    if ret:
        cv2.imwrite("captured_image.png", frame)  # Save the captured image
        cap.release()  # Release the camera
        global img_path
        img_path = "captured_image.png"
        process_image()
    else:
        messagebox.showerror("Error", "Could not capture an image.")
        cap.release()

# Function to perform edge detection, measurements, and display total width and height on an image
def process_image():
    if img_path:
        # Read image
        image = cv2.imread(img_path)

        # Perform edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.Canny(blurred, 50, 100)

        # Find contours
        cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Sort contours from left to right
        (cnts, _) = contours.sort_contours(cnts)

        # Reference object dimensions (modify as needed)
        ref_object = cnts[0]
        box = cv2.minAreaRect(ref_object)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        dist_in_pixel = euclidean(tl, tr)
        dist_in_cm = 2
        pixel_per_cm = dist_in_pixel / dist_in_cm

        # Process contours
        measurements = []

        for cnt in cnts:
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            wid = euclidean(tl, tr) / pixel_per_cm
            ht = euclidean(tr, br) / pixel_per_cm
            measurements.append((wid, ht))

            # Draw contours and measurements
            cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
            mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
            mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))
            cv2.putText(image, "{:.1f}cm x {:.1f}cm".format(wid, ht),
                        (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Calculate total width and height
        total_width = sum(wid for wid, _ in measurements)
        total_height = max(ht for _, ht in measurements)

        # Display the total values at the top of the displayed image
        cv2.putText(image, "Total Width: {:.1f}cm".format(total_width),
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image, "Total Height: {:.1f}cm".format(total_height),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Show the image with measurements and total values
        cv2.imshow("Processed Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Function to perform object detection using AI
def perform_object_detection():
    global model
    if img_path and model:
        # Load and preprocess the image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(image_rgb)

        # Run inference on the model
        detections = model(input_tensor[tf.newaxis, ...])

        # Process detection results and draw bounding boxes
        # Modify this code based on your specific model's output format
        # ...

# Add buttons for image capture, image selection, and object detection
capture_button = tk.Button(root, text="Capture Image", command=capture_image, width=20, height=3)
capture_button.pack(pady=10)

select_button = tk.Button(root, text="Select Image", command=select_image, width=20, height=3)
select_button.pack(pady=10)

load_model_button = tk.Button(root, text="Load AI Model", command=load_model, width=20, height=3)
load_model_button.pack(pady=10)

detect_objects_button = tk.Button(root, text="Detect Objects", command=perform_object_detection, width=20, height=3)
detect_objects_button.pack(pady=10)

# Information label for users
info_label = tk.Label(root, text="Click buttons to perform actions.", bg="#f0f0f0", pady=10)
info_label.pack()

# Run the GUI main loop
root.mainloop()
