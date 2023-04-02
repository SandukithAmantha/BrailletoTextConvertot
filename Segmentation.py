import numpy as np
import pickle
import cv2
from keras.models import load_model
import os
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
from scipy.signal import find_peaks

labelDictionary = {0: '0', 1: 'අ', 2: 'ඉ', 3: 'ඊ', 4: 'උ', 5: 'එ', 6: 'ඒ', 7: 'ඔ', 8: 'ක', 9: 'ක්', 10: 'කා',
                      11: 'කැ', 12: 'කෑ', 13: 'කි', 14: 'කී', 15: 'කු', 16: 'කූ', 17: 'කෙ', 18: 'කේ', 19: 'කො',
                      20: 'කෝ', 21: 'ඛ', 22: 'ග', 23: 'ගි', 24: 'ගී', 25: 'ගු', 26: 'ගූ', 27: 'ඝ', 28: 'ඟ', 29: 'ච',
                      30: 'ඡ', 31: 'ජ', 32: 'ජ්', 33: 'ජි', 34: 'ජී', 35: 'ඣ', 36: 'ඤ', 37: 'ඥ', 38: 'ට', 39: 'ඨ',
                      40: 'ඩ',
                      41: 'ඪ', 42: 'ණ', 43: 'ඬ', 44: 'ත', 45: 'ත්', 46: 'ථ', 47: 'ථි', 48: 'ථී', 49: 'ද', 50: 'දු',
                      51: 'දූ', 52: 'ධ', 53: 'න', 54: 'ඳ', 55: 'ප', 56: 'පු'}

###parameterrs###
width = 640
height = 480
threshold = 0.65

# Load the saved model
model = load_model('model.h5')
print(model.input_shape)

# Function to perform pre-processing on the image
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
    threshold = ThresholdFilter(gray, T=150)
    # Normalize image pixels between 0 and 255
    normalized = normalize(threshold)
    # Convert image to uint8 type
    normalized = normalized.astype(np.uint8)
    return normalized

# Function to perform character segmentation using vertical projection
def segment_characters(img):
    _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # Find continuous white pixels
    peaks = []
    predicted_labels = []
    prev_pixel = 0
    white_dot_count = 0  # initialize white dot count to 0
    for i in range(thresh.shape[1]):
        col_sum = np.sum(thresh[:, i])
        if col_sum > 0 and prev_pixel == 0:
            peaks.append(i)
            white_dot_count += 1  # increment white dot count
        prev_pixel = col_sum

        if white_dot_count == 2:  # if two white dots are found
            letter = img[:, peaks[0]:peaks[1]]
            _, letter = cv2.threshold(letter, 100, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(letter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
                biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
                x, y, w, h = cv2.boundingRect(biggest_contour)
                # draw bounding box around character
                cv2.rectangle(img, (peaks[0] + x, y), (peaks[0] + x + w, y + h), (255, 0, 0), 2)
                letter_image = letter[y:y + h, x:x + w]
                letter_image = cv2.resize(letter_image, (32, 32))
                letter_image = letter_image.reshape((1, 32, 32, 1))
                letter_image = letter_image.astype('float32')
                letter_image /= 255
                predictedLabel = np.argmax(model.predict(letter_image))
                predicted_labels.append(predictedLabel)
            # reset white dot count and peaks
            white_dot_count = 0
            peaks = []

    # display the image with bounding boxes
    cv2.imshow("Segmented Characters", img)
    cv2.waitKey(0)
    return predicted_labels


# Read the input image file
#imgOriginal = cv2.imread("TestingImages/MageNamaNavindu.jpeg")

print('Image Preprocessing started')

# Load image
img = cv2.imread('TestingImages/MageNamaNavindu2.jpg')

# Preprocess image
preprocessed = preprocessImage(img)

img = Image.fromarray(np.uint8(preprocessed))

# Invert the colors
inverted_img = ImageOps.invert(img)

# Convert the inverted image back to a NumPy array
inverted_array = np.array(inverted_img)

#------------

#imgPreProcess = preProcessing(inverted_img)
cv2.imshow("PreProcess Window", inverted_array)
cv2.waitKey(0)
print('Preprocessing done')

# Segment characters from the input image
predicted_labels = segment_characters(inverted_array)

# Map predicted labels to characters using the label dictionary
predicted_text = ''.join([labelDictionary[label] for label in predicted_labels])

# Print the predicted text
print(predicted_text)

# Display the image with the predicted text
cv2.putText(inverted_array, predicted_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
cv2.imshow("Testing Window", inverted_array)
cv2.waitKey(0)
