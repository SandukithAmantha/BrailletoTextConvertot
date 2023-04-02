
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

from main import resize_shape, preprocess_image

'''
# Load the trained model
model = load_model('model.h5')

# Read the image and invert its colors
img = cv2.imread('TestingImages/MageNamaNavindu.jpeg', cv2.IMREAD_GRAYSCALE)
inverted_img = cv2.bitwise_not(img)

# Resize the image to the expected input shape of the model
resized_img = cv2.resize(inverted_img, (32, 32))
input_img = np.expand_dims(resized_img, axis=-1)
input_img = np.expand_dims(input_img, axis=0)

cv2.imshow("Testing Window1", inverted_img)
cv2.waitKey(0)
'''

model = load_model('model.h5')

input_image = cv2.imread("TestingImages/2.jpg",cv2.IMREAD_COLOR)
resized_character = cv2.resize(input_image,resize_shape)
input_preprocessed_image = preprocess_image(resized_character)
reshaped_character = input_preprocessed_image.reshape(1,resize_shape[0],resize_shape[1],1)
prediction=model.predict(reshaped_character,verbose=0)
predicted_class=np.argmax(prediction,axis=1)

# # Use the loaded model to make predictions
# predictions = load_model.predict(reshaped_character)
# print("predicted class: ",predicted_class)
# predicted_class=np.argmax(prediction,axis=1)

#%matplotlib inline
plt.axis('off')
plt.imshow(input_image, cmap='gray')
plt.show()
print("predicted class: ",predicted_class)