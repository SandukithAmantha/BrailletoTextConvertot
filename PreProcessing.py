from tkinter import Image

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps

def RGBtoGrayScale(img):
    # If the image is already grayscale, return it
    if img.ndim == 2:
        return img
    # If the image is RGB, convert to grayscale
    size = img.shape
    for i in range(size[0]):
        for j in range(size[1]):
            img[i, j, 0] = img[i, j, 0] * 0.2989 + img[i, j, 1] * 0.5870 + img[i, j, 2] * 0.1140
    img = img[:, :, 0].squeeze()
    return img

# Function to apply threshold filter to image
def ThresholdFilter(img, T=1):
    size = img.shape
    for i in range(size[0]):
        for j in range(size[1]):
            img[i, j] = 0 if img[i, j] < T else 1
    return img

# Function to normalize image pixels between 0 and 255
def normalize(img):
    imin = np.min(img)
    imax = np.max(img)
    return (((img - imin)/(imax - imin)) * 255).astype(np.float64)

# Function to preprocess image
def preprocessImage(img):
    # Convert image to grayscale
    gray = RGBtoGrayScale(img)
    # Apply threshold filter to grayscale image
    threshold = ThresholdFilter(gray, T=175)
    # Normalize image pixels between 0 and 255
    normalized = normalize(threshold)
    # Convert image to uint8 type
    normalized = normalized.astype(np.uint8)
    return normalized

# Load image
img = cv2.imread('TestingImages/aaa.jpeg')

# Preprocess image
preprocessed = preprocessImage(img)

img = Image.fromarray(np.uint8(preprocessed))

# Invert the colors
inverted_img = ImageOps.invert(img)

# Convert the inverted image back to a NumPy array
inverted_array = np.array(inverted_img)

# Display preprocessed image
plt.imshow(inverted_img, cmap='gray')
plt.show()

cv2.imshow("PreProcessed Window", inverted_img)
cv2.waitKey(0)